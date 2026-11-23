use crate::lm::Lm;
use serde::Serialize;
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
    pub preds: bool,
    /// Counts for a word to be a target of auto-correction, and to be taken as
    /// a real word when typed.
    pub known: u64,
    pub valid: u64,
}

#[derive(Clone)]
struct Hyp {
    chars: Vec<char>,
    row: usize,
    lm: f32,
    sp: f32,
    next: Rc<Vec<f32>>,
}

#[derive(Serialize)]
pub struct Word {
    pub typed: String,
    pub commit: String,
    pub cands: Vec<String>,
    /// The best known word, how far it scores above the typed word, and
    /// whether the typed word counts as real: to tune the thresholds offline.
    pub best: String,
    pub margin: f32,
    pub valid: bool,
}

#[derive(Serialize)]
pub struct Sentence {
    pub out: String,
    pub words: Vec<Word>,
    pub preds: Vec<Vec<String>>,
}

pub struct Decoder {
    pub lm: Lm,
    tok: HashMap<char, u16>,
    chr: Vec<Option<char>>,
    lang: u16,
    keys: Vec<Key>,
    variants: HashMap<char, Vec<char>>,
    lex: HashMap<String, u64>,
    p: Params,
    /// The context: log-probabilities after it, where it starts in the text, what it holds.
    root_next: Rc<Vec<f32>>,
    from: usize,
    fed: Vec<char>,
    hyps: Vec<Hyp>,
    /// Seconds per tap: (letter, time) for each tap of every sentence.
    pub lat: Vec<(bool, f64)>,
    /// Seconds per next-word prediction, which a keyboard runs in the background after a separator.
    pub bg: Vec<f64>,
}

const SWAPPED: &str = ".,!?;:)…";

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
        vocab: &[String],
        lang: &str,
        keys: Vec<Key>,
        variants: HashMap<char, Vec<char>>,
        lex: HashMap<String, u64>,
        p: Params,
    ) -> Result<Self, String> {
        if vocab.len() != lm.vocab_size()
            || p.beam != lm.rows()
            || [p.sx, p.sy, p.ssy]
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.0)
            || [
                p.mx, p.my, p.smy, p.stray, p.variant, p.omit, p.oov, p.tau, p.tau_word,
            ]
            .iter()
            .any(|x| !x.is_finite())
            || keys.iter().any(|k| !k.cx.is_finite() || !k.cy.is_finite())
            || p.space.is_some_and(|(x0, x1, y)| {
                !x0.is_finite() || !x1.is_finite() || !y.is_finite() || x0 >= x1
            })
        {
            return Err("Invalid decoder vocabulary or parameters".into());
        }
        let mut chr = Vec::with_capacity(vocab.len());
        let mut tok = HashMap::new();
        for (i, token) in vocab.iter().enumerate() {
            let mut chars = token.chars();
            let ch = chars.next().filter(|_| chars.next().is_none());
            chr.push(ch);
            if let Some(c) = ch
                && tok.insert(c, i as u16).is_some()
            {
                return Err("Duplicate vocabulary character".into());
            }
        }
        let marker = format!("<{lang}>");
        let lang = vocab
            .iter()
            .position(|t| t == &marker)
            .ok_or("Language token is missing")? as u16;
        Ok(Self {
            lm,
            tok,
            chr,
            lang,
            keys,
            variants,
            lex,
            p,
            root_next: Rc::new(vec![]),
            from: 0,
            fed: vec![],
            hyps: vec![],
            lat: vec![],
            bg: vec![],
        })
    }

    fn t(&self, c: char) -> u16 {
        *self.tok.get(&c).unwrap_or(&1)
    }

    /// Brings the context to the language token plus `text`, or the end of
    /// it that fits in half the context when it would leave no room for a word.
    fn sync(&mut self, text: &str) -> Result<(), String> {
        let want: Vec<char> = text.chars().collect();
        let same = !self.root_next.is_empty()
            && want
                .get(self.from..)
                .is_some_and(|tail| tail.starts_with(&self.fed));
        let new = if same {
            want.len() - self.from - self.fed.len()
        } else {
            want.len()
        };
        if !same || self.lm.room() < new + self.lm.suffix() {
            self.from = want.len() - want.len().min(self.lm.ctx() / 2);
            self.lm.reset();
            self.root_next = Rc::new(self.lm.feed(self.lang)?);
            self.fed.clear();
        }
        for &c in want.iter().skip(self.from + self.fed.len()) {
            self.root_next = Rc::new(self.lm.feed(self.t(c))?);
            self.fed.push(c);
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.fed.clear();
        self.root_next = Rc::new(vec![]);
        self.sync("\n")
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
            let score = -0.5 * z * z;
            if score.is_finite() {
                v.push((' ', score));
            }
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
                    .ok_or("Missing parent hypothesis")
            })
            .collect::<Result<_, _>>()?;
        let mut free: Vec<usize> = (0..self.p.beam).filter(|r| !parents.contains(r)).collect();
        let mut claimed = HashSet::new();
        let mut next = vec![];
        let mut copies = vec![];
        for (pi, c, lm, sp) in kids {
            let parent = self.hyps.get(pi).ok_or("Missing parent hypothesis")?;
            let row = if claimed.insert(parent.row) {
                parent.row
            } else {
                let r = free.pop().ok_or("No free beam row")?;
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
                    h.next = Rc::new(out.next().ok_or("Missing beam prediction")?);
                }
                Ok::<_, String>(h)
            })
            .collect::<Result<_, _>>()?;
        Ok(())
    }

    fn start(&mut self) -> Result<(), String> {
        self.lm.clear(0)?;
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
    fn omissions(&mut self, spaces: bool) -> Result<(), String> {
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
                    kids.push((i, self.chr.get(t).copied().flatten(), lm, h.sp));
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

    fn tap(&mut self, c: char, x: i32, y: i32) -> Result<(), String> {
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
                        .get(usize::from(self.t(ch)))
                        .ok_or("Missing token probability")?,
                    h.sp + l,
                ));
            }
            kids.push((i, None, h.lm, h.sp + self.p.stray));
        }
        kids.sort_by(|a, b| (b.2 + b.3).total_cmp(&(a.2 + a.3)));
        let mut seen = HashSet::new();
        let mut unique = Vec::with_capacity(self.p.beam);
        for k in kids {
            let mut text = self
                .hyps
                .get(k.0)
                .ok_or("Missing parent hypothesis")?
                .chars
                .clone();
            text.extend(k.1);
            if seen.insert(text) {
                unique.push(k);
                if unique.len() == self.p.beam {
                    break;
                }
            }
        }
        self.advance(unique)
    }

    fn finish(&mut self, typed: &str, auto: bool, sep: char) -> Result<Word, String> {
        let st = self.t(sep);
        let mut scored: Vec<(String, f32, usize)> = self
            .hyps
            .iter()
            .enumerate()
            .filter(|(_, h)| !h.chars.is_empty())
            .map(|(i, h)| {
                let w: String = h.chars.iter().collect();
                let oov = if self.known(&w) { 0.0 } else { self.p.oov };
                Ok::<_, String>((
                    w,
                    h.lm + h.sp
                        + h.next
                            .get(usize::from(st))
                            .ok_or("Missing separator probability")?
                        + oov,
                    i,
                ))
            })
            .collect::<Result<_, _>>()?;
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let lit = scored
            .iter()
            .find(|s| s.0 == typed)
            .map(|s| s.1)
            .unwrap_or(f32::NEG_INFINITY);
        let valid = self.valid(typed);
        let tau = if valid { self.p.tau_word } else { self.p.tau };
        let top = scored.iter().find(|s| self.known(&s.0));
        let (best, margin) = top
            .map(|b| (b.0.clone(), (b.1 - lit).min(99.0)))
            .unwrap_or_default();
        let commit = match top {
            Some(b) if auto && b.0 != typed && b.1 - lit > tau => b.0.clone(),
            _ => typed.to_string(),
        };
        if let Some(s) = scored.iter().find(|s| s.0 == commit) {
            let hyp = self.hyps.get(s.2).ok_or("Missing selected hypothesis")?;
            if hyp.chars.len() <= self.lm.suffix() && self.lm.commit(hyp.row)? {
                self.root_next = hyp.next.clone();
                self.fed.extend(commit.chars());
            }
        }
        let cands = scored
            .iter()
            .map(|s| s.0.clone())
            .filter(|w| w != typed && self.known(w))
            .take(18)
            .collect();
        Ok(Word {
            typed: typed.to_string(),
            commit,
            cands,
            best: if auto { best } else { String::new() },
            margin,
            valid,
        })
    }

    /// The three likeliest next words after the root's text.
    fn predict(&mut self) -> Result<Vec<String>, String> {
        if !self.p.preds {
            return Ok(vec![]);
        }
        self.start()?;
        let mut done: Vec<(String, f32)> = vec![];
        let seps = [' ', '.', ',', '!', '?', '\n'];
        for _ in 0..14 {
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
                    .is_some_and(|(word, kid)| word.1 > kid.2)
            {
                break;
            }
            self.advance(kids)?;
        }
        Ok(done.into_iter().take(3).map(|d| d.0).collect())
    }

    /// One sentence of taps, typed the way Ime.kt types them.
    pub fn sentence(&mut self, taps: &[(char, i32, i32)]) -> Result<Sentence, String> {
        // The text starts as a new unit of the training data does.
        let (mut text, mut word, mut typed) = (String::from("\n"), false, String::new());
        let (mut words, mut preds) = (vec![], vec![]);
        let mut weak = false;
        self.reset()?;
        let mut last: Option<(std::time::Instant, bool)> = None;
        for &(c, x, y) in taps {
            if let Some((t, l)) = last {
                self.lat.push((l, t.elapsed().as_secs_f64()));
            }
            last = Some((std::time::Instant::now(), c != ' ' && c.is_alphanumeric()));
            let connector = (c == '\'' || c == '-') && word;
            if c != ' ' && (c.is_alphanumeric() || connector) {
                if !word {
                    let t = std::time::Instant::now();
                    self.sync(&text)?;
                    let p = self.predict()?;
                    preds.push(p);
                    let bg = t.elapsed();
                    self.bg.push(bg.as_secs_f64());
                    if let Some((t0, _)) = &mut last {
                        *t0 += bg;
                    }
                    self.start()?;
                    word = true;
                }
                typed.push(c);
                self.tap(c, x, y)?;
                weak = false;
                continue;
            }
            let auto = c == ' ' || !c.is_alphanumeric();
            if word {
                if self.p.omit < 0.0 {
                    self.omissions(false)?;
                }
                let w = self.finish(&typed, auto, c)?;
                text.push_str(&w.commit);
                words.push(w);
                word = false;
                typed.clear();
                if c == ' ' {
                    text.push(' ');
                    weak = true;
                    continue;
                }
            }
            if c == ' ' {
                text.push(' ');
                weak = text.chars().rev().nth(1).is_some_and(char::is_alphanumeric);
            } else if weak && SWAPPED.contains(c) && text.ends_with(' ') {
                text.pop();
                text.push(c);
                text.push(' ');
                weak = false;
            } else {
                text.push(c);
                weak = false;
            }
        }
        if word {
            let w = self.finish(&typed, false, ' ')?;
            text.push_str(&w.commit);
            words.push(w);
        }
        if let Some((t, l)) = last {
            self.lat.push((l, t.elapsed().as_secs_f64()));
        }
        Ok(Sentence {
            out: text.trim().to_string(),
            words,
            preds,
        })
    }
}
