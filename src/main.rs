#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use d3::decode::{Decoder, Key, Params};
use d3::lm::{self, Lm, Model};
use npusvc_sdk::lrt::{CPU, Lrt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "decoder --libs DIR --model DIR --layouts FILE --lang L --lexicon FILE --taps FILE --out FILE
  [--beam 16] [--ctx 192] [--suffix 48] [--wq8 0|1] [--i8 0|1] [--preds 1] [--limit N] [--touch ../data/taps.json]
  [--stray -7] [--variant -1] [--omit 0|-4.6] [--oov -4] [--tau 4] [--tau-word 3] [--known 3] [--valid 100]
decoder --model DIR --export-keyboard DIR --lexicon FILE [--ctx 192] [--beam 16] [--wq8 0|1]";

type Result<T> = std::result::Result<T, String>;

struct Args(HashMap<String, String>);

impl Args {
    fn parse() -> Result<Self> {
        let mut flags = HashMap::new();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let key = arg
                .strip_prefix("--")
                .ok_or_else(|| format!("Expected a flag: {arg}"))?;
            if ![
                "libs",
                "model",
                "layouts",
                "lang",
                "lexicon",
                "taps",
                "out",
                "beam",
                "ctx",
                "suffix",
                "wq8",
                "i8",
                "preds",
                "limit",
                "touch",
                "stray",
                "variant",
                "omit",
                "oov",
                "tau",
                "tau-word",
                "known",
                "valid",
                "export-keyboard",
            ]
            .contains(&key)
            {
                return Err(format!("Unknown flag --{key}"));
            }
            let value = args
                .next()
                .filter(|value| !value.starts_with("--") && !value.is_empty())
                .ok_or_else(|| format!("Missing value for --{key}"))?;
            if flags.insert(key.into(), value).is_some() {
                return Err(format!("Duplicate flag --{key}"));
            }
        }
        Ok(Self(flags))
    }

    fn get<'a>(&'a self, name: &str, default: &'a str) -> &'a str {
        self.0.get(name).map(String::as_str).unwrap_or(default)
    }

    fn require(&self, name: &str) -> Result<&str> {
        self.0
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("Missing --{name}"))
    }

    fn number<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T> {
        self.0.get(name).map_or(Ok(default), |s| {
            s.parse()
                .map_err(|_| format!("Invalid number for --{name}"))
        })
    }

    fn toggle(&self, name: &str, default: bool) -> Result<bool> {
        match self.0.get(name).map(String::as_str) {
            None => Ok(default),
            Some("0") => Ok(false),
            Some("1") => Ok(true),
            Some(_) => Err(format!("Expected 0 or 1 for --{name}")),
        }
    }
}

#[derive(Deserialize)]
struct Layout {
    lang: String,
    width: f32,
    keys: Vec<(i32, f32, f32, f32, f32)>,
    #[serde(default)]
    more: HashMap<char, String>,
}

#[derive(Deserialize)]
struct Offset {
    mean: [f32; 2],
    cov: [[f32; 2]; 2],
}

#[derive(Deserialize)]
struct TouchKeys {
    #[serde(rename = "SPACE")]
    space: Offset,
}

#[derive(Deserialize)]
struct Touch {
    keyboard_px: [f32; 2],
    key_px: HashMap<String, [f32; 2]>,
    all_letters: Offset,
    keys: TouchKeys,
}

#[derive(Deserialize)]
struct Taps {
    taps: Vec<(u32, i32, i32)>,
}

#[derive(Serialize)]
struct KeyboardConfig {
    layers: usize,
    ctx: usize,
}

fn load_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T> {
    let data = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_slice(&data).map_err(|e| format!("{path}: {e}"))
}

fn parse_count(line: &str) -> Result<(&str, u64)> {
    let (word, count) = line
        .split_once('\t')
        .filter(|(word, _)| !word.is_empty())
        .ok_or("Expected a tab-separated word and count")?;
    Ok((word, count.parse().map_err(|_| "Invalid lexicon count")?))
}

fn lexicon(path: &str, variants: &HashMap<char, Vec<char>>) -> Result<HashMap<String, u64>> {
    let data = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut words = HashMap::new();
    for (i, line) in data.lines().enumerate() {
        let (word, count) = parse_count(line).map_err(|e| format!("{path}:{}: {e}", i + 1))?;
        words.insert(word.to_owned(), count);
    }
    // A more frequent accented spelling can distinguish valid words from
    // diacritics dropped in the subtitles used to build the lexicon.
    let mut base = HashMap::new();
    let mut sorted: Vec<_> = variants.iter().collect();
    sorted.sort_by_key(|(b, _)| **b);
    for (&b, vs) in sorted {
        for &v in vs {
            base.insert(v, b);
        }
    }
    let mut accented = HashMap::<String, u64>::new();
    for (word, &count) in &words {
        let plain: String = word
            .chars()
            .map(|c| base.get(&c).copied().unwrap_or(c))
            .collect();
        if plain != *word {
            let value = accented.entry(plain).or_default();
            *value = (*value).max(count);
        }
    }
    let mut dropped = 0;
    for (word, count) in &mut words {
        if accented.get(word).is_some_and(|&n| n / 10 >= *count) {
            *count = 0;
            dropped += 1;
        }
    }
    eprintln!(
        "lexicon: {} words, {dropped} without diacritics dropped",
        words.len()
    );
    Ok(words)
}

type LayoutParams = (Vec<Key>, HashMap<char, Vec<char>>, Params);

fn load_layout(args: &Args, lang: &str, beam: usize) -> Result<LayoutParams> {
    let layouts: Vec<Layout> = load_json(args.require("layouts")?)?;
    let layout = layouts
        .into_iter()
        .find(|l| l.lang == lang)
        .ok_or("Language layout is missing")?;
    let mut space = None;
    let mut keys = Vec::new();
    for (code, x, y, w, h) in layout.keys {
        if ![x, y, w, h].iter().all(|v| v.is_finite()) || w <= 0.0 || h <= 0.0 {
            return Err("Invalid key rectangle".into());
        }
        // Negative Android key codes are actions rather than characters.
        if code < 0 {
            continue;
        }
        let code = char::from_u32(code as u32).ok_or("Invalid layout codepoint")?;
        let cy = y + h / 2.0;
        if code == ' ' {
            space = Some((x, x + w, cy));
        } else if code.is_alphanumeric() || code == '\'' {
            keys.push(Key {
                code,
                cx: x + w / 2.0,
                cy,
            });
        }
    }
    let variants = layout
        .more
        .into_iter()
        .map(|(c, s)| (c, s.chars().collect()))
        .collect();
    let touch: Touch = load_json(args.get("touch", "../data/taps.json"))?;
    let [touch_width, _] = touch.keyboard_px;
    if !layout.width.is_finite()
        || layout.width <= 0.0
        || !touch_width.is_finite()
        || touch_width <= 0.0
    {
        return Err("Invalid keyboard width".into());
    }
    let scale = layout.width / touch_width;
    let &[kw, kh] = touch
        .key_px
        .get("a")
        .ok_or("Missing letter size in touch data")?;
    let &[_, sh] = touch
        .key_px
        .get("SPACE")
        .ok_or("Missing space size in touch data")?;
    if ![kw, kh, sh].iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err("Invalid key size in touch data".into());
    }
    let Offset {
        mean: [mx, my],
        cov: [[xx, _], [_, yy]],
    } = touch.all_letters;
    let Offset {
        mean: [_, smy],
        cov: [_, [_, syy]],
    } = touch.keys.space;
    Ok((
        keys,
        variants,
        Params {
            beam,
            mx: mx * kw * scale,
            my: my * kh * scale,
            sx: xx.sqrt() * kw * scale,
            sy: yy.sqrt() * kh * scale,
            space,
            smy: smy * sh * scale,
            ssy: syy.sqrt() * sh * scale,
            stray: args.number("stray", -7.0)?,
            variant: args.number("variant", -1.0)?,
            omit: args.number("omit", 0.0)?,
            oov: args.number("oov", -4.0)?,
            tau: args.number("tau", 4.0)?,
            tau_word: args.number("tau-word", 3.0)?,
            preds: args.toggle("preds", true)?,
            known: args.number("known", 3)?,
            valid: args.number("valid", 100)?,
        },
    ))
}

fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let filename = path
        .file_name()
        .ok_or("Missing output filename")?
        .to_string_lossy();
    let tmp = path.with_file_name(format!(".{filename}.{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    if let Err(error) = file.write_all(data).and_then(|()| fs::rename(&tmp, path)) {
        let cleanup = fs::remove_file(&tmp);
        if let Err(e) = cleanup {
            return Err(format!(
                "{}: {error}; cleaning {}: {e}",
                path.display(),
                tmp.display()
            ));
        }
        return Err(format!("{}: {error}", path.display()));
    }
    Ok(())
}

fn export_keyboard(
    model: &Model,
    dir: &Path,
    lex: &str,
    ctx: usize,
    beam: usize,
    wq8: bool,
) -> Result<()> {
    let rows = beam.checked_add(1).ok_or("Invalid keyboard batch size")?;
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let step = lm::build_keyboard(model, ctx, rows, wq8)?;
    let root = lm::build_keyboard(model, ctx, 1, wq8)?;
    let vocab = serde_json::to_vec(model.vocabulary()).map_err(|e| e.to_string())?;
    let config = serde_json::to_vec(&KeyboardConfig {
        layers: model.layers(),
        ctx,
    })
    .map_err(|e| e.to_string())?;
    let data = fs::read_to_string(lex).map_err(|e| format!("{lex}: {e}"))?;
    let mut filtered = String::new();
    for (i, line) in data.lines().enumerate() {
        let (_, count) = parse_count(line).map_err(|e| format!("{lex}:{}: {e}", i + 1))?;
        if count >= 3 {
            filtered.push_str(line);
            filtered.push('\n');
        }
    }
    atomic_write(&dir.join("step.tflite"), &step)?;
    atomic_write(&dir.join("root.tflite"), &root)?;
    atomic_write(&dir.join("emb.bin"), &model.embedding_bytes()?)?;
    atomic_write(&dir.join("vocab.json"), &vocab)?;
    atomic_write(&dir.join("config.json"), &config)?;
    atomic_write(&dir.join("lexicon.tsv"), filtered.as_bytes())?;
    Ok(())
}

fn run(args: Args) -> Result<()> {
    let path = args.require("model")?;
    let beam = args.number("beam", 16)?;
    let ctx = args.number("ctx", 192)?;
    let wq8 = args.toggle("wq8", false)?;
    let model = Model::load(path)?;
    if let Some(out) = args.0.get("export-keyboard") {
        return export_keyboard(
            &model,
            Path::new(out),
            args.require("lexicon")?,
            ctx,
            beam,
            wq8,
        );
    }

    let (libs, lang) = (args.require("libs")?, args.require("lang")?);
    let (lex, taps, output) = (
        args.require("lexicon")?,
        args.require("taps")?,
        args.require("out")?,
    );
    let suffix = args.number("suffix", 48)?;
    let limit = args.number("limit", usize::MAX)?;
    model.validate_graph(ctx, suffix, beam)?;
    let (keys, variants, params) = load_layout(&args, lang, beam)?;
    let lex = lexicon(lex, &variants)?;
    let acts = args
        .toggle("i8", false)?
        .then(|| lm::load_acts(Path::new(path)))
        .transpose()?;
    let q = match (acts.is_some(), wq8) {
        (true, true) => "i8q8",
        (true, false) => "i8",
        (false, true) => "q8",
        (false, false) => "",
    };
    let graph = |rows, size| -> Result<PathBuf> {
        let file = Path::new(path).join(format!("lm3_b{rows}_c{ctx}_s{size}{q}.tflite"));
        if !file
            .try_exists()
            .map_err(|e| format!("{}: {e}", file.display()))?
        {
            atomic_write(
                &file,
                &lm::build(&model, acts.as_ref(), ctx, size, rows, wq8)?,
            )?;
        }
        Ok(file)
    };
    let runtime = Lrt::new(libs, libs, None).map_err(|e| format!("LiteRT: {e}"))?;
    let compile = |file: PathBuf| {
        let path = file.to_str().ok_or("Invalid graph path")?;
        runtime
            .compile(path, CPU)
            .map_err(|e| format!("{path}: {e}"))
    };
    let lm = Lm::new(
        compile(graph(beam, suffix)?)?,
        compile(graph(1, 1)?)?,
        &model,
        ctx,
        suffix,
        beam,
    )?;
    let mut decoder = Decoder::new(lm, model.vocabulary(), lang, keys, variants, lex, params)?;

    let input = BufReader::new(File::open(taps).map_err(|e| format!("{taps}: {e}"))?);
    let mut out = BufWriter::new(File::create(output).map_err(|e| format!("{output}: {e}"))?);
    let start = std::time::Instant::now();
    let mut count = 0;
    for (i, line) in input.lines().take(limit).enumerate() {
        let tap_error = |e: &dyn std::fmt::Display| format!("{taps}:{}: {e}", i + 1);
        let line = line.map_err(|e| tap_error(&e))?;
        let sentence: Taps =
            serde_json::from_str(&line).map_err(|_| tap_error(&"Invalid taps document"))?;
        let taps = sentence
            .taps
            .into_iter()
            .map(|(c, x, y)| {
                Ok((
                    char::from_u32(c).ok_or_else(|| tap_error(&"Invalid tap codepoint"))?,
                    x,
                    y,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let result = decoder.sentence(&taps)?;
        serde_json::to_writer(&mut out, &result).map_err(|e| format!("{output}: {e}"))?;
        writeln!(out).map_err(|e| format!("{output}: {e}"))?;
        count += 1;
    }
    out.flush().map_err(|e| format!("{output}: {e}"))?;
    let lat = |letter| {
        decoder
            .lat
            .iter()
            .filter(|p| p.0 == letter)
            .map(|p| p.1 * 1e3)
            .collect::<Vec<_>>()
    };
    for (name, mut values) in [
        ("letters", lat(true)),
        ("separators", lat(false)),
        ("next words", decoder.bg.iter().map(|x| x * 1e3).collect()),
    ] {
        values.sort_by(f64::total_cmp);
        if let Some(&max) = values.last() {
            let q = |p| {
                values
                    .get(((values.len() - 1) as f64 * p) as usize)
                    .copied()
                    .unwrap_or(max)
            };
            eprintln!(
                "{name}: {} taps, p50 {:.1} ms, p95 {:.1} ms, max {max:.1} ms",
                values.len(),
                q(0.5),
                q(0.95)
            );
        }
    }
    eprintln!(
        "{lang}: {count} sentences in {:.1} s, {} steps, {:.2} ms each",
        start.elapsed().as_secs_f64(),
        decoder.lm.steps,
        decoder.lm.secs * 1e3 / decoder.lm.steps.max(1) as f64
    );
    Ok(())
}

fn main() -> ExitCode {
    if std::env::args()
        .skip(1)
        .any(|arg| arg == "--help" || arg == "-h")
    {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match Args::parse().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
