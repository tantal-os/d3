use super::lm::Lm;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub struct Key {
    pub code: char,
    pub cx: f32,
    pub cy: f32,
}

pub struct Params {
    pub beam: usize,
    /// Touch offset mean and spread in pixels of the layout.
    pub mx: f32,
    pub my: f32,
    pub sx: f32,
    pub sy: f32,
    /// The space bar: left and right edge, centre y, and its touch offset and
    /// spread in y. A tap above it may be a space that hit a letter.
    pub space: Option<(f32, f32, f32)>,
    pub smy: f32,
    pub ssy: f32,
    /// Log-probabilities of a stray tap and of a long-press variant typed as its base letter.
    pub stray: f32,
    pub variant: f32,
    /// Log-probability of a skipped character; 0 turns the search for them off.
    pub omit: f32,
    /// Log penalty for a word outside the lexicon.
    pub oov: f32,
    /// Margin over the typed word to auto-correct, when it is not a word and when it is.
    pub tau: f32,
    pub tau_word: f32,
    /// Counts for a word to be a target of auto-correction, and to be taken as
    /// a real word when typed.
    pub known: u64,
    pub valid: u64,
}

impl Params {
    /// The bench's touch model (research/typing/data/taps.json: Google's TSI
    /// taps on a 1440 px wide keyboard, in key units) scaled to a keyboard
    /// `width` px wide, and the thresholds tuned on the bench.
    pub fn new(width: f32, space: Option<(f32, f32, f32)>) -> Params {
        let (kw, kh, sh) = (
            134.0 * width / 1440.0,
            206.0 * width / 1440.0,
            208.0 * width / 1440.0,
        );
        Params {
            beam: 16,
            mx: -0.012286451 * kw,
            my: -0.058705036 * kh,
            sx: 0.076027828f32.sqrt() * kw,
            sy: 0.03783205f32.sqrt() * kh,
            space,
            smy: -0.207544 * sh,
            ssy: 0.053_944_1_f32.sqrt() * sh,
            stray: -7.0,
            variant: -1.0,
            omit: -4.6,
            oov: -4.0,
            tau: 3.0,
            tau_word: 3.0,
            known: 3,
            valid: 100,
        }
    }
}

#[derive(Clone)]
struct Hyp {
    chars: Vec<char>,
    row: usize,
    lm: f32,
    sp: f32,
    next: Rc<Vec<f32>>,
}

pub struct Word {
    pub commit: String,
    pub cands: Vec<String>,
    /// Whether the typed word counts as real.
    pub valid: bool,
}

pub struct Decoder {
    lm: Lm,
    root: Lm,
    tok: HashMap<char, u16>,
    chr: Vec<Option<char>>,
    lang: u16,
    keys: Vec<Key>,
    variants: HashMap<char, Vec<char>>,
    lex: HashMap<String, u64>,
    p: Params,
    root_next: Rc<Vec<f32>>,
    fed: Vec<char>,
    hyps: Vec<Hyp>,
}

impl Decoder {
    /// Every word of `w` (a hypothesis may hold a space that hit a letter).
    fn known(&self, w: &str) -> bool {
        w.split(' ')
            .all(|w| self.lex.get(w).is_some_and(|&n| n >= self.p.known))
    }

    fn valid(&self, w: &str) -> bool {
        self.lex.get(w).is_some_and(|&n| n >= self.p.valid)
    }

    pub fn new(
        lm: Lm,
        root: Lm,
        vocab: &[String],
        lang: &str,
        lex: HashMap<String, u64>,
    ) -> Result<Self, String> {
        let (keys, variants, p) = (vec![], HashMap::new(), Params::new(1440.0, None));
        if vocab.len() > usize::from(u16::MAX) + 1 {
            return Err("vocabulary too large".into());
        }
        let chr: Vec<_> = vocab
            .iter()
            .map(|t| {
                let mut chars = t.chars();
                chars.next().filter(|_| chars.next().is_none())
            })
            .collect();
        let tok = chr
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|c| (c, i as u16)))
            .collect();
        let tag = format!("<{lang}>");
        let lang = vocab
            .iter()
            .position(|t| t == &tag)
            .ok_or("missing language token")? as u16;
        Ok(Decoder {
            lm,
            root,
            tok,
            chr,
            lang,
            keys,
            variants,
            lex,
            p,
            root_next: Rc::new(vec![]),
            fed: vec![],
            hyps: vec![],
        })
    }

    fn t(&self, c: char) -> u16 {
        *self.tok.get(&c).unwrap_or(&1)
    }

    /// Brings the root to the language token plus `text`.
    pub fn sync(&mut self, text: &str) -> Result<(), String> {
        let want: Vec<char> = text.chars().collect();
        if self.root_next.is_empty() || !want.starts_with(&self.fed) {
            self.root.reset_root()?;
            self.root_next = Rc::new(
                self.root
                    .step(&[(0, self.lang)])?
                    .into_iter()
                    .next()
                    .ok_or("missing root output")?,
            );
            self.fed.clear();
        }
        for &c in want.iter().skip(self.fed.len()) {
            let t = self.t(c);
            self.root_next = Rc::new(
                self.root
                    .step(&[(0, t)])?
                    .into_iter()
                    .next()
                    .ok_or("missing root output")?,
            );
        }
        self.fed = want;
        Ok(())
    }

    pub fn set_layout(&mut self, keys: Vec<Key>, variants: HashMap<char, Vec<char>>, p: Params) {
        (self.keys, self.variants, self.p) = (keys, variants, p);
    }

    /// log P(key | tap) over the keys near the tap.
    fn near(&self, x: f32, y: f32) -> Vec<(char, f32)> {
        let mut v: Vec<(char, f32)> = self
            .keys
            .iter()
            .map(|k| {
                let zx = (x - k.cx - self.p.mx) / self.p.sx;
                let zy = (y - k.cy - self.p.my) / self.p.sy;
                (k.code, -0.5 * (zx * zx + zy * zy))
            })
            .filter(|(_, l)| *l > -8.0)
            .collect();
        if let Some((x0, x1, cy)) = self.p.space
            && x >= x0
            && x < x1
        {
            let z = (y - cy - self.p.smy) / self.p.ssy;
            v.push((' ', -0.5 * z * z));
        }
        let m = v.iter().map(|x| x.1).fold(f32::MIN, f32::max);
        let s = v.iter().map(|x| (x.1 - m).exp()).sum::<f32>().ln() + m;
        v.iter_mut().for_each(|x| x.1 -= s);
        v
    }

    /// Moves to the next generation: `kids` as (parent index, new char), the
    /// rows reassigned so that no parent's cache is overwritten before it is copied.
    fn advance(&mut self, kids: Vec<(usize, Option<char>, f32, f32)>) -> Result<(), String> {
        let parents: HashSet<usize> = kids
            .iter()
            .map(|k| {
                self.hyps
                    .get(k.0)
                    .map(|h| h.row)
                    .ok_or("missing beam parent")
            })
            .collect::<Result<_, _>>()?;
        let mut free: Vec<usize> = (0..self.p.beam).filter(|r| !parents.contains(r)).collect();
        let mut claimed = HashSet::new();
        let mut next = vec![];
        let mut copies = vec![];
        for (pi, c, lm, sp) in kids {
            let parent = self.hyps.get(pi).ok_or("missing beam parent")?;
            let row = if claimed.insert(parent.row) {
                parent.row
            } else {
                let r = free.pop().ok_or("no free beam row")?;
                copies.push((parent.row, r));
                r
            };
            let mut chars = parent.chars.clone();
            chars.extend(c);
            next.push((
                Hyp {
                    chars,
                    row,
                    lm,
                    sp,
                    next: parent.next.clone(),
                },
                c,
            ));
        }
        for (from, to) in copies {
            self.lm.copy(from, to)?;
        }
        let rows: Vec<(usize, u16)> = next
            .iter()
            .filter_map(|(h, c)| c.map(|c| (h.row, self.t(c))))
            .collect();
        let mut out = if rows.is_empty() {
            vec![]
        } else {
            self.lm.step(&rows)?
        }
        .into_iter();
        self.hyps = next
            .into_iter()
            .map(|(mut h, c)| {
                if c.is_some() {
                    h.next = Rc::new(out.next().ok_or("missing beam output")?);
                }
                Ok(h)
            })
            .collect::<Result<_, String>>()?;
        Ok(())
    }

    pub fn start(&mut self) -> Result<(), String> {
        self.lm.import_root(&self.root)?;
        self.hyps = vec![Hyp {
            chars: vec![],
            row: 0,
            lm: 0.0,
            sp: 0.0,
            next: self.root_next.clone(),
        }];
        Ok(())
    }

    /// Before a tap or a separator: hypotheses that also carry a character typed without a
    /// tap, from the likeliest next letters of the best hypotheses.
    pub fn omissions(&mut self, spaces: bool) -> Result<(), String> {
        let best = self
            .hyps
            .iter()
            .map(|h| h.lm + h.sp)
            .fold(f32::MIN, f32::max);
        let mut kids: Vec<_> = self
            .hyps
            .iter()
            .enumerate()
            .map(|(i, h)| (i, None, h.lm, h.sp))
            .collect();
        for (i, h) in self.hyps.iter().enumerate().take(4) {
            let space = spaces && h.chars.last().is_some_and(|&c| c != ' ');
            let mut top: Vec<(usize, f32)> = h
                .next
                .iter()
                .cloned()
                .enumerate()
                .filter(|(t, _)| {
                    self.chr
                        .get(*t)
                        .copied()
                        .flatten()
                        .is_some_and(|c| c.is_alphabetic() || (space && c == ' '))
                })
                .collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            for &(t, lp) in top.iter().take(2) {
                let lm = h.lm + lp + self.p.omit;
                if lm + h.sp > best - 10.0 {
                    kids.push((
                        i,
                        *self.chr.get(t).ok_or("invalid vocabulary index")?,
                        lm,
                        h.sp,
                    ));
                }
            }
        }
        if kids.len() == self.hyps.len() {
            return Ok(());
        }
        kids.sort_by(|a, b| (b.2 + b.3).total_cmp(&(a.2 + a.3)));
        kids.truncate(self.p.beam);
        self.advance(kids)
    }

    pub fn tap(&mut self, c: char, x: i32, y: i32) -> Result<(), String> {
        if self.p.omit < 0.0 && x >= 0 && c.is_alphanumeric() {
            self.omissions(true)?;
        }
        // A digit tap may be a letter of the top row that landed high.
        let opts: Vec<(char, f32)> = if x < 0 || !c.is_alphanumeric() {
            vec![(c, 0.0)]
        } else {
            let mut o = vec![];
            for (k, l) in self.near(x as f32, y as f32) {
                o.push((k, l));
                for v in self.variants.get(&k).into_iter().flatten() {
                    o.push((*v, l + self.p.variant));
                }
            }
            o
        };
        let mut kids = vec![];
        for (i, h) in self.hyps.iter().enumerate() {
            for &(ch, l) in &opts {
                if ch == ' ' && h.chars.last().is_none_or(|&c| c == ' ') {
                    continue;
                }
                kids.push((
                    i,
                    Some(ch),
                    h.lm + *h
                        .next
                        .get(self.t(ch) as usize)
                        .ok_or("missing token score")?,
                    h.sp + l,
                ));
            }
            kids.push((i, None, h.lm, h.sp + self.p.stray));
        }
        kids.sort_by(|a, b| (b.2 + b.3).total_cmp(&(a.2 + a.3)));
        let mut seen = HashSet::new();
        let mut unique = Vec::new();
        for kid in kids {
            let mut chars = self
                .hyps
                .get(kid.0)
                .ok_or("missing beam parent")?
                .chars
                .clone();
            chars.extend(kid.1);
            if seen.insert(chars) {
                unique.push(kid);
                if unique.len() == self.p.beam {
                    break;
                }
            }
        }
        self.advance(unique)
    }

    pub fn finish(&self, typed: &str, auto: bool, sep: char) -> Result<Word, String> {
        let st = self.t(sep);
        let mut scored: Vec<(String, f32)> = self
            .hyps
            .iter()
            .filter(|h| !h.chars.is_empty())
            .map(|h| {
                let w: String = h.chars.iter().collect();
                let oov = if self.known(&w) { 0.0 } else { self.p.oov };
                Ok((
                    w,
                    h.lm + h.sp + h.next.get(st as usize).ok_or("missing separator score")? + oov,
                ))
            })
            .collect::<Result<_, String>>()?;
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let lit = scored
            .iter()
            .find(|s| s.0 == typed)
            .map(|s| s.1)
            .unwrap_or(f32::NEG_INFINITY);
        let valid = self.valid(typed);
        let tau = if valid { self.p.tau_word } else { self.p.tau };
        let top = scored.iter().find(|s| self.known(&s.0));
        let commit = match top {
            Some(b) if auto && b.0 != typed && b.1 - lit > tau => b.0.clone(),
            _ => typed.to_string(),
        };
        let cands = scored
            .iter()
            .map(|s| s.0.clone())
            .filter(|w| w != typed && self.known(w))
            .take(18)
            .collect();
        Ok(Word {
            commit,
            cands,
            valid,
        })
    }

    /// The three likeliest next words after the root's text; empty once `stop` says so.
    pub fn predict(&mut self, stop: impl Fn() -> bool) -> Result<Vec<String>, String> {
        self.start()?;
        let mut done: Vec<(String, f32)> = vec![];
        let seps = [' ', '.', ',', '!', '?', '\n'];
        for _ in 0..14 {
            if stop() {
                return Ok(vec![]);
            }
            let mut kids = vec![];
            for (i, h) in self.hyps.iter().enumerate() {
                let mut best: Vec<(usize, f32)> = h.next.iter().cloned().enumerate().collect();
                best.sort_by(|a, b| b.1.total_cmp(&a.1));
                for &(t, lp) in best.iter().take(self.p.beam) {
                    let Some(c) = self.chr.get(t).copied().flatten() else {
                        continue;
                    };
                    if seps.contains(&c) {
                        if !h.chars.is_empty() {
                            let w: String = h.chars.iter().collect();
                            if self.known(&w) {
                                done.push((w, h.lm + lp));
                            }
                        }
                    } else if c.is_alphanumeric() || c == '\'' || c == '-' {
                        kids.push((i, Some(c), h.lm + lp, 0.0));
                    }
                }
            }
            kids.sort_by(|a, b| b.2.total_cmp(&a.2));
            kids.truncate(self.p.beam);
            done.sort_by(|a, b| b.1.total_cmp(&a.1));
            done.dedup_by(|a, b| a.0 == b.0);
            if kids.is_empty()
                || done
                    .get(2)
                    .zip(kids.first())
                    .is_some_and(|(word, next)| word.1 > next.2)
            {
                break;
            }
            self.advance(kids)?;
        }
        Ok(done.into_iter().take(3).map(|d| d.0).collect())
    }
}
