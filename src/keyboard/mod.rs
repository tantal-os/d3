mod decode;
mod lm;

use decode::{Decoder, Params};
pub use decode::{Key, Word};
use lm::Lm;
use npusvc_sdk::lrt::Compiled;
use std::collections::HashMap;
use std::rc::Rc;

pub type Tap = (char, i32, i32);
const MAX_CONTEXT: usize = 120;
const KEEP_CONTEXT: usize = 60;
const MAX_TAPS: usize = 32;

pub struct Session {
    decoder: Decoder,
    full: String,
    start: usize,
    word: Option<(String, Vec<Tap>)>,
}

pub struct ModelAssets<'a> {
    pub vocabulary: &'a [u8],
    pub embeddings: &'a [u8],
    pub lexicon: &'a [u8],
    pub language: &'a str,
    pub layers: usize,
    pub context: usize,
}

impl Session {
    pub fn open(step: Compiled, root: Compiled, assets: ModelAssets<'_>) -> Result<Self, String> {
        let vocabulary: Vec<String> =
            serde_json::from_slice(assets.vocabulary).map_err(|_| "invalid vocabulary")?;
        let (weights, trailing) = assets.embeddings.as_chunks::<4>();
        if !trailing.is_empty() {
            return Err("truncated embedding".into());
        }
        let embeddings: Rc<[f32]> = weights.iter().map(|w| f32::from_le_bytes(*w)).collect();
        let mut lexicon = HashMap::new();
        for line in std::str::from_utf8(assets.lexicon)
            .map_err(|_| "invalid lexicon encoding")?
            .lines()
        {
            let (word, count) = line.split_once('\t').ok_or("invalid lexicon entry")?;
            lexicon.insert(
                word.to_owned(),
                count.parse().map_err(|_| "invalid word count")?,
            );
        }
        let batch = Params::new(1440.0, None).beam + 1;
        let step = Lm::new(
            step,
            embeddings.clone(),
            vocabulary.len(),
            assets.layers,
            assets.context,
            batch,
        )?;
        let root = Lm::new(
            root,
            embeddings,
            vocabulary.len(),
            assets.layers,
            assets.context,
            1,
        )?;
        Ok(Self {
            decoder: Decoder::new(step, root, &vocabulary, assets.language, lexicon)?,
            full: String::new(),
            start: 0,
            word: None,
        })
    }

    pub fn layout(
        &mut self,
        keys: Vec<Key>,
        variants: HashMap<char, Vec<char>>,
        width: f32,
        space: Option<(f32, f32, f32)>,
    ) -> Result<(), String> {
        if !width.is_finite() || width <= 0.0 {
            return Err("invalid layout width".into());
        }
        self.decoder
            .set_layout(keys, variants, Params::new(width, space));
        self.word = None;
        Ok(())
    }

    fn window(&mut self, before: &str) -> String {
        let chars: Vec<_> = before.chars().collect();
        if !before.starts_with(&self.full) || chars.len().saturating_sub(self.start) > MAX_CONTEXT {
            self.start = 0;
            if chars.len() > MAX_CONTEXT {
                let from = chars.len() - KEEP_CONTEXT;
                self.start = chars
                    .iter()
                    .skip(from)
                    .position(|&c| c == ' ')
                    .map_or(from, |i| from + i + 1);
            }
        }
        self.full = before.to_owned();
        let mut text = String::new();
        if self.start == 0 {
            text.push('\n');
        }
        text.extend(chars.iter().skip(self.start));
        text
    }

    fn apply(&mut self, before: &str, taps: &[Tap]) -> Result<(), String> {
        let window = self.window(before);
        self.decoder.sync(&window)?;
        let done = match &self.word {
            Some((at, previous)) if *at == window && taps.starts_with(previous) => previous.len(),
            _ => {
                self.decoder.start()?;
                0
            }
        };
        for &(c, x, y) in taps.iter().skip(done) {
            self.decoder.tap(c, x, y)?;
        }
        self.word = Some((window, taps.to_owned()));
        Ok(())
    }

    pub fn suggest(&mut self, before: &str, taps: &[Tap]) -> Result<Option<Word>, String> {
        if taps.is_empty() || taps.len() > MAX_TAPS {
            return Ok(None);
        }
        self.apply(before, taps)?;
        self.decoder
            .finish(&taps.iter().map(|t| t.0).collect::<String>(), true, ' ')
            .map(Some)
    }

    pub fn commit(
        &mut self,
        before: &str,
        taps: &[Tap],
        separator: char,
    ) -> Result<String, String> {
        let typed = taps.iter().map(|t| t.0).collect::<String>();
        if taps.is_empty() || taps.len() > MAX_TAPS {
            return Ok(typed);
        }
        self.apply(before, taps)?;
        self.decoder.omissions(false)?;
        self.word = None;
        Ok(self.decoder.finish(&typed, true, separator)?.commit)
    }

    pub fn predict(
        &mut self,
        before: &str,
        cancelled: impl Fn() -> bool,
    ) -> Result<Vec<String>, String> {
        let window = self.window(before);
        self.decoder.sync(&window)?;
        self.word = None;
        self.decoder.predict(cancelled)
    }
}
