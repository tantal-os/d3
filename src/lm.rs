use npusvc_sdk::graphs::{ADD, DIV, EXP, FULLY_CONNECTED, MUL, SUM};
use npusvc_sdk::lrt::Compiled;
use npusvc_sdk::tfl::{Dt, F, Net, T};
use npusvc_sdk::weights::Weights;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

pub const DH: usize = 64;

/// Largest |input| and |output| of each linear layer on test text (train/export.py).
pub type Acts = HashMap<String, (f32, f32)>;

pub fn load_acts(model: &Path) -> Result<Acts, String> {
    let file = model.join("act_absmax.txt");
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    let mut acts = Acts::new();
    for (line, value) in text.lines().enumerate() {
        let fields: Vec<_> = value.split_whitespace().collect();
        let [name, input, output] = fields.as_slice() else {
            return Err(format!("Invalid activation range at line {}", line + 1));
        };
        let range = |s: &str| -> Result<f32, String> {
            s.parse::<f32>()
                .ok()
                .filter(|x| x.is_finite() && *x > 0.0)
                .ok_or_else(|| format!("Invalid activation range for {name}"))
        };
        if acts
            .insert(name.to_string(), (range(input)?, range(output)?))
            .is_some()
        {
            return Err(format!("Duplicate activation range for {name}"));
        }
    }
    if acts.is_empty() {
        return Err("Empty activation ranges".into());
    }
    Ok(acts)
}

#[derive(Deserialize)]
struct Config {
    layers: usize,
}

pub struct Model {
    weights: Weights,
    layers: usize,
    width: usize,
    vocab: Vec<String>,
}

impl Model {
    pub fn load(dir: &str) -> Result<Self, String> {
        let read = |name| {
            let path = Path::new(dir).join(name);
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
        };
        let cfg: Config =
            serde_json::from_str(&read("config.json")?).map_err(|e| format!("config.json: {e}"))?;
        let vocab: Vec<String> =
            serde_json::from_str(&read("vocab.json")?).map_err(|e| format!("vocab.json: {e}"))?;
        let weights = Weights::load(dir)?;
        let (shape, _) = weights.get("emb.weight")?;
        let [entries, width] = shape.as_slice() else {
            return Err("Expected a two-dimensional embedding".into());
        };
        let width = usize::try_from(*width).map_err(|e| e.to_string())?;
        if usize::try_from(*entries).ok() != Some(vocab.len())
            || !(2..=usize::from(u16::MAX) + 1).contains(&vocab.len())
            || width == 0
            || !width.is_multiple_of(DH)
        {
            return Err("Invalid embedding shape or vocabulary length".into());
        }
        if cfg.layers == 0 || cfg.layers > 256 {
            return Err("Invalid layer count".into());
        }
        let check = |name: &str, required: &[i32]| -> Result<(), String> {
            let (shape, _) = weights.get(name)?;
            if shape != required {
                return Err(format!("Unexpected tensor shape for {name}"));
            }
            Ok(())
        };
        let d = width as i32;
        let norm = |name: &str| -> Result<(), String> {
            check(&format!("{name}.weight"), &[d])?;
            check(&format!("{name}.bias"), &[d])
        };
        norm("ln")?;
        for l in 0..cfg.layers {
            let p = format!("layers.{l}");
            norm(&format!("{p}.ln1"))?;
            norm(&format!("{p}.ln2"))?;
            let (up, _) = weights.get(&format!("{p}.up.weight"))?;
            let [ff, _] = up.as_slice() else {
                return Err(format!("Unexpected tensor shape for {p}.up.weight"));
            };
            for (name, input, output) in [
                ("q", d, d),
                ("k", d, d),
                ("v", d, d),
                ("o", d, d),
                ("up", d, *ff),
                ("down", *ff, d),
            ] {
                check(&format!("{p}.{name}.weight"), &[output, input])?;
                check(&format!("{p}.{name}.bias"), &[output])?;
            }
        }
        Ok(Self {
            weights,
            layers: cfg.layers,
            width,
            vocab,
        })
    }

    pub fn vocabulary(&self) -> &[String] {
        &self.vocab
    }

    pub fn embedding_bytes(&self) -> Result<Vec<u8>, String> {
        Ok(self
            .weights
            .get("emb.weight")?
            .1
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect())
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn validate_graph(&self, ctx: usize, suf: usize, rows: usize) -> Result<(), String> {
        let bytes = ctx
            .max(suf)
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(self.width))
            .and_then(|x| x.checked_mul(self.layers))
            .and_then(|x| x.checked_mul(4));
        if ctx < 4
            || suf == 0
            || suf > ctx
            || rows == 0
            || bytes.is_none_or(|x| x > 512 * 1024 * 1024)
            || rows
                .checked_mul(self.vocab.len())
                .is_none_or(|x| x > 16 * 1024 * 1024)
        {
            return Err("Invalid context, suffix or batch size".into());
        }
        Ok(())
    }
}

/// x @ W.T + b: float, or with `acts` int8 inputs, weights (per row) and
/// outputs between QUANTIZE and DEQUANTIZE.
fn dense(
    n: &mut Net,
    acts: Option<&Acts>,
    x: i32,
    pre: &str,
    (ws, wv, bv): Layer<'_>,
    scale: f32,
) -> Result<i32, String> {
    let wv: Vec<f32> = wv.iter().map(|x| x * scale).collect();
    let bv: Vec<f32> = bv.iter().map(|x| x * scale).collect();
    let mut s = n.shape(x);
    let [out, _] = ws else {
        return Err(format!("Expected a matrix for {pre}"));
    };
    *s.last_mut()
        .ok_or_else(|| format!("Missing input shape for {pre}"))? = *out;
    // The input of `down` (after GELU) has outliers and the head's int8 steps
    // of 0.2 nats reorder close words: both stay float, the rest of the
    // decoder's output then matches float32 on 39 of 40 dev sentences.
    let int8 = pre != "head" && !pre.ends_with(".down");
    let Some(&(ai, ao)) = acts.filter(|_| int8).and_then(|a| a.get(pre)) else {
        let wt = if n.wq8 {
            n.const_q8_rows(ws, &wv)
        } else {
            n.const_f32(ws, &wv)
        };
        let bt = n.const_f32(&[*out], &bv);
        return Ok(n.fc_with(x, (wt, bt), -8.0, 8.0));
    };
    let xs = n.shape(x);
    let xq = n.act_i8(&xs, ai);
    n.op(114, &[x], &[xq], None);
    let (wt, sc) = n.q8_rows(ws, &wv);
    let si = n.scale_of(xq);
    let bs: Vec<f32> = sc.iter().map(|s| s * si).collect();
    let bt = n.q32_bias(&bv, &bs);
    let yq = n.act_i8(&s, ao * scale);
    n.op(
        FULLY_CONNECTED,
        &[xq, wt, bt],
        &[yq],
        Some((8, T(vec![(0, F::B(0)), (2, F::B(1))]))),
    );
    let y = n.act_f32(&s);
    n.op(6, &[yq], &[y], None);
    Ok(y)
}

type Layer<'a> = (&'a [i32], &'a [f32], &'a [f32]);

fn layer<'a>(w: &'a Weights, pre: &str) -> Result<Layer<'a>, String> {
    let (shape, weights) = w.get(&format!("{pre}.weight"))?;
    Ok((shape, weights, &w.get(&format!("{pre}.bias"))?.1))
}

// Rotary positions on [B*H, 64]: x * cos + rot(x) * sin, rot swapping each
// pair (a, b) to (-b, a) as a constant matrix, cos and sin given per row and
// head. Kept 2-D: the TPU compiler rejects the reshape after a 3-D FC
// ("Reshape input/output slice mismatch").
fn rope(n: &mut Net, x: i32, cs: i32, sn: i32) -> i32 {
    let mut p = vec![0.0f32; DH * DH];
    for j in 0..DH / 2 {
        if let Some(cell) = p.get_mut((2 * j) * DH + 2 * j + 1) {
            *cell = -1.0;
        }
        if let Some(cell) = p.get_mut((2 * j + 1) * DH + 2 * j) {
            *cell = 1.0;
        }
    }
    let pw = n.const_f32(&[DH as i32, DH as i32], &p);
    let pb = n.const_f32(&[DH as i32], &[0.0; DH]);
    let r = n.fc_with(x, (pw, pb), -8.0, 8.0);
    let a = n.binary(MUL, x, cs, -8.0, 8.0);
    let b = n.binary(MUL, r, sn, -8.0, 8.0);
    n.binary(ADD, a, b, -8.0, 8.0)
}

pub fn build(
    model: &Model,
    acts: Option<&Acts>,
    ctx: usize,
    suf: usize,
    batch: usize,
    wq8: bool,
) -> Result<Vec<u8>, String> {
    build_graph(model, acts, ctx, Some(suf), batch, wq8)
}

pub fn build_keyboard(
    model: &Model,
    ctx: usize,
    batch: usize,
    wq8: bool,
) -> Result<Vec<u8>, String> {
    build_graph(model, None, ctx, None, batch, wq8)
}

fn build_graph(
    model: &Model,
    acts: Option<&Acts>,
    ctx: usize,
    suf: Option<usize>,
    batch: usize,
    wq8: bool,
) -> Result<Vec<u8>, String> {
    model.validate_graph(ctx, suf.unwrap_or(1), batch)?;
    let (ctx, batch, suf) = (ctx as i32, batch as i32, suf.map(|s| s as i32));
    let w = &model.weights;
    let layers = model.layers;
    let mut n = Net::new(Dt::F32);
    n.wq8 = wq8;
    let (_, ev) = w.get("emb.weight")?;
    let (vocab, d) = (model.vocab.len() as i32, model.width as i32);
    let heads = d / DH as i32;
    let (bh, dh) = (batch * heads, DH as i32);
    let mut x = n.input(&[batch, d], -8.0, 8.0);
    let cs = n.input(&[bh, dh], -1.0, 1.0);
    let sn = n.input(&[bh, dh], -1.0, 1.0);
    let (mask, suffix_mask, caches) = if let Some(s) = suf {
        let mask = n.input(&[heads, batch, ctx], -1e9, 0.0);
        let suffix_mask = n.input(&[bh, 1, s], -1e9, 0.0);
        let caches = (0..layers)
            .map(|_| {
                let k = n.input(&[heads, dh, ctx], -8.0, 8.0);
                let v = n.input(&[heads, ctx, dh], -8.0, 8.0);
                let ks = n.input(&[bh, dh, s], -8.0, 8.0);
                let vs = n.input(&[bh, s, dh], -8.0, 8.0);
                (k, v, Some((ks, vs)))
            })
            .collect::<Vec<_>>();
        (mask, Some(suffix_mask), caches)
    } else {
        let mask = n.input(&[bh, 1, ctx + 1], -1e9, 0.0);
        let caches = (0..layers)
            .map(|_| {
                let k = n.input(&[bh, dh, ctx], -8.0, 8.0);
                let v = n.input(&[bh, ctx, dh], -8.0, 8.0);
                (k, v, None)
            })
            .collect();
        (mask, None, caches)
    };
    let mut new = vec![];
    for (l, &(kc, vc, suffix_cache)) in caches.iter().enumerate() {
        let p = format!("layers.{l}");
        let h = n.layer_norm_with(x, w, &format!("{p}.ln1"))?;
        let fc = |n: &mut Net, x: i32, name: &str, scale: f32| {
            let pre = format!("{p}.{name}");
            dense(n, acts, x, &pre, layer(w, &pre)?, scale)
        };
        let q = fc(&mut n, h, "q", 1.0 / (DH as f32).sqrt())?;
        let q = n.reshape(q, &[bh, dh]);
        let q = rope(&mut n, q, cs, sn);
        let k = fc(&mut n, h, "k", 1.0)?;
        let k = n.reshape(k, &[bh, dh]);
        let k = rope(&mut n, k, cs, sn);
        let k = n.reshape(k, &[bh, dh, 1]);
        let v = fc(&mut n, h, "v", 1.0)?;
        let v = n.reshape(v, &[bh, 1, dh]);
        new.extend([k, v]);
        let o = if let (Some((ks, vs)), Some(suffix_mask)) = (suffix_cache, suffix_mask) {
            // Softmax over context, own characters and the new position, written
            // out (exp, sum, divide) so that no cache is ever concatenated.

            let qh = n.reshape(q, &[batch, heads, dh]);
            let qh = n.transpose(qh, &[1, 0, 2]);
            let sc = n.bmm(qh, kc, -16.0, 16.0);
            let sc = n.binary(ADD, sc, mask, -16.0, 16.0);
            let ec = n.unary(EXP, sc, 0.0, 1e6);
            let zc = n.reduce(SUM, ec, 0.0, 1e6);
            let oc = n.bmm(ec, vc, -1e6, 1e6);
            let oc = n.transpose(oc, &[1, 0, 2]);
            let oc = n.reshape(oc, &[bh, 1, dh]);
            let zc = n.transpose(zc, &[1, 0, 2]);
            let zc = n.reshape(zc, &[bh, 1, 1]);
            let q3 = n.reshape(q, &[bh, 1, dh]);
            let ss = n.bmm(q3, ks, -16.0, 16.0);
            let ss = n.binary(ADD, ss, suffix_mask, -16.0, 16.0);
            let es = n.unary(EXP, ss, 0.0, 1e6);
            let zs = n.reduce(SUM, es, 0.0, 1e6);
            let os = n.bmm(es, vs, -1e6, 1e6);
            let en = n.bmm(q3, k, -16.0, 16.0);
            let en = n.unary(EXP, en, 0.0, 1e6);
            let on = n.binary(MUL, v, en, -1e6, 1e6);
            let o = n.binary(ADD, oc, os, -1e6, 1e6);
            let o = n.binary(ADD, o, on, -1e6, 1e6);
            let z = n.binary(ADD, zc, zs, 0.0, 1e6);
            let z = n.binary(ADD, z, en, 0.0, 1e6);
            n.binary(DIV, o, z, -8.0, 8.0)
        } else {
            let q = n.reshape(q, &[bh, 1, dh]);
            let keys = n.concat_any(&[kc, k], 2);
            let values = n.concat_any(&[vc, v], 1);
            let scores = n.bmm(q, keys, -16.0, 16.0);
            let scores = n.binary(ADD, scores, mask, -16.0, 16.0);
            let weights = n.softmax(scores);
            n.bmm(weights, values, -8.0, 8.0)
        };
        let o = n.reshape(o, &[batch, d]);
        let o = fc(&mut n, o, "o", 1.0)?;
        x = n.binary(ADD, x, o, -8.0, 8.0);
        let h = n.layer_norm_with(x, w, &format!("{p}.ln2"))?;
        let h = fc(&mut n, h, "up", 1.0)?;
        let h = n.gelu(h, false);
        let h = fc(&mut n, h, "down", 1.0)?;
        x = n.binary(ADD, x, h, -8.0, 8.0);
    }
    let y = n.layer_norm_with(x, w, "ln")?;
    let logits = dense(
        &mut n,
        acts,
        y,
        "head",
        (&[vocab, d], ev, &vec![0.0; vocab as usize]),
        1.0,
    )?;
    n.output(logits);
    new.into_iter().for_each(|t| n.output(t));
    n.build()
}

fn cache(layer: usize) -> usize {
    5 + 4 * layer
}

fn row<T>(values: &[T], r: usize, width: usize) -> Result<&[T], String> {
    values
        .get(r * width..(r + 1) * width)
        .ok_or_else(|| "Invalid tensor row".into())
}

fn row_mut<T>(values: &mut [T], r: usize, width: usize) -> Result<&mut [T], String> {
    values
        .get_mut(r * width..(r + 1) * width)
        .ok_or_else(|| "Invalid tensor row".into())
}

struct Graph {
    compiled: Compiled,
    suffix: usize,
    len: Vec<usize>,
}

struct Meta {
    ctx: usize,
    vocab: usize,
    width: usize,
    heads: usize,
    layers: usize,
    embedding: Vec<f32>,
    inv: Vec<f32>,
}

struct Updates {
    key: Vec<Vec<f32>>,
    value: Vec<Vec<f32>>,
}

struct Run {
    probabilities: Vec<Vec<f32>>,
    updates: Vec<Updates>,
    seconds: f64,
}

impl Graph {
    fn open(compiled: Compiled, m: &Meta, rows: usize, suffix: usize) -> Result<Self, String> {
        let (b, d, h, ctx, s, dh, v) = (
            rows as i32,
            m.width as i32,
            m.heads as i32,
            m.ctx as i32,
            suffix as i32,
            DH as i32,
            m.vocab as i32,
        );
        let bh = b * h;
        let mut inputs = vec![
            vec![b, d],
            vec![bh, dh],
            vec![bh, dh],
            vec![h, b, ctx],
            vec![bh, 1, s],
        ];
        let mut outputs = vec![vec![b, v]];
        for _ in 0..m.layers {
            inputs.extend([
                vec![h, dh, ctx],
                vec![h, ctx, dh],
                vec![bh, dh, s],
                vec![bh, s, dh],
            ]);
            outputs.extend([vec![bh, dh, 1], vec![bh, 1, dh]]);
        }
        if compiled.input_count() != inputs.len() || compiled.output_count() != outputs.len() {
            return Err("Unexpected tensor count in decoder graph".into());
        }
        let check = |shape: &[i32], spec: &npusvc_sdk::lrt::TensorSpec| -> Result<(), String> {
            let bytes = shape.iter().map(|&x| x as usize).product::<usize>() * 4;
            if spec.shape != shape || spec.element_type != 1 || spec.bytes != bytes {
                return Err("Unexpected tensor type or shape in decoder graph".into());
            }
            Ok(())
        };
        for (i, shape) in inputs.iter().enumerate() {
            check(shape, compiled.input_spec(i)?)?;
            // A NaN in a masked cache slot can still poison the value products.
            compiled.write_input(i, |b| b.fill(0))?;
        }
        for (i, shape) in outputs.iter().enumerate() {
            check(shape, compiled.output_spec(i)?)?;
        }
        Ok(Self {
            compiled,
            suffix,
            len: vec![0; rows],
        })
    }

    fn run(&mut self, m: &Meta, clen: usize, rows: &[(usize, u16)]) -> Result<Run, String> {
        let (d, h, ctx, s) = (m.width, m.heads, m.ctx, self.suffix);
        if clen > ctx
            || rows
                .iter()
                .any(|&(r, t)| r >= self.len.len() || usize::from(t) >= m.vocab)
        {
            return Err("Invalid decoder step".into());
        }
        self.compiled.write_input_f32(0, |buf| {
            for &(r, t) in rows {
                row_mut(buf, r, d)?.copy_from_slice(row(&m.embedding, usize::from(t), d)?);
            }
            Ok::<_, String>(())
        })??;
        for (i, sin) in [(1, false), (2, true)] {
            self.compiled.write_input_f32(i, |buf| {
                for (chunk, &len) in buf.chunks_exact_mut(h * DH).zip(&self.len) {
                    let p = (clen + len) as f32;
                    let mut angles = [0.0; DH / 2];
                    for (angle, inv) in angles.iter_mut().zip(&m.inv) {
                        *angle = if sin {
                            (p * inv).sin()
                        } else {
                            (p * inv).cos()
                        };
                    }
                    for head in chunk.chunks_exact_mut(DH) {
                        for (pair, &angle) in head.chunks_exact_mut(2).zip(&angles) {
                            pair.fill(angle);
                        }
                    }
                }
            })?;
        }
        self.compiled.write_input_f32(3, |buf| {
            for (i, x) in buf.iter_mut().enumerate() {
                *x = if i % ctx < clen { 0.0 } else { -1e9 };
            }
        })?;
        self.compiled.write_input_f32(4, |buf| {
            for (chunk, &len) in buf.chunks_exact_mut(h * s).zip(&self.len) {
                for (i, x) in chunk.iter_mut().enumerate() {
                    *x = if i % s < len { 0.0 } else { -1e9 };
                }
            }
        })?;
        let start = std::time::Instant::now();
        self.compiled.run()?;
        let seconds = start.elapsed().as_secs_f64();
        let outputs = self.compiled.outputs()?;
        let pick = |i: usize, width: usize| -> Result<Vec<Vec<f32>>, String> {
            let tensor = outputs.get(i).ok_or("Missing decoder output")?;
            rows.iter()
                .map(|&(r, _)| {
                    row(tensor, r, width * 4)?
                        .chunks_exact(4)
                        .map(|bytes| {
                            let bytes: [u8; 4] =
                                bytes.try_into().map_err(|_| "Invalid float output")?;
                            let v = f32::from_ne_bytes(bytes);
                            if v.is_finite() {
                                Ok(v)
                            } else {
                                Err("Non-finite decoder output".into())
                            }
                        })
                        .collect()
                })
                .collect()
        };
        let mut probabilities = pick(0, m.vocab)?;
        for z in &mut probabilities {
            let max = z.iter().copied().fold(f32::MIN, f32::max);
            let sum = z.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
            z.iter_mut().for_each(|x| *x = (*x - max) - sum);
        }
        let updates = (0..m.layers)
            .map(|l| {
                Ok::<_, String>(Updates {
                    key: pick(1 + 2 * l, h * DH)?,
                    value: pick(2 + 2 * l, h * DH)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Run {
            probabilities,
            updates,
            seconds,
        })
    }

    fn put_context(
        &self,
        m: &Meta,
        l: usize,
        pos: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), String> {
        self.compiled.write_input_f32(cache(l), |buf| {
            for (cell, &k) in buf.iter_mut().skip(pos).step_by(m.ctx).zip(key) {
                *cell = k;
            }
        })?;
        self.compiled.write_input_f32(cache(l) + 1, |buf| {
            for (head, v) in buf.chunks_exact_mut(m.ctx * DH).zip(value.chunks_exact(DH)) {
                row_mut(head, pos, DH)?.copy_from_slice(v);
            }
            Ok::<_, String>(())
        })??;
        Ok(())
    }
}

pub struct Lm {
    beam: Graph,
    feed: Graph,
    m: Meta,
    suf: usize,
    len: usize,
    pub steps: usize,
    pub secs: f64,
}

impl Lm {
    pub fn new(
        beam: Compiled,
        feed: Compiled,
        model: &Model,
        ctx: usize,
        suf: usize,
        rows: usize,
    ) -> Result<Self, String> {
        model.validate_graph(ctx, suf, rows)?;
        let (_, embedding) = model.weights.get("emb.weight")?;
        let m = Meta {
            ctx,
            vocab: model.vocab.len(),
            width: model.width,
            heads: model.width / DH,
            layers: model.layers,
            embedding: embedding.clone(),
            inv: (0..DH / 2)
                .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / DH as f32))
                .collect(),
        };
        Ok(Self {
            beam: Graph::open(beam, &m, rows, suf)?,
            feed: Graph::open(feed, &m, 1, 1)?,
            m,
            suf,
            len: 0,
            steps: 0,
            secs: 0.0,
        })
    }

    pub fn rows(&self) -> usize {
        self.beam.len.len()
    }

    pub fn suffix(&self) -> usize {
        self.suf
    }

    pub fn vocab_size(&self) -> usize {
        self.m.vocab
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    pub fn ctx(&self) -> usize {
        self.m.ctx
    }

    pub fn room(&self) -> usize {
        self.m.ctx - self.len
    }

    pub fn clear(&mut self, r: usize) -> Result<(), String> {
        *self.beam.len.get_mut(r).ok_or("Invalid beam row")? = 0;
        Ok(())
    }

    pub fn copy(&mut self, from: usize, to: usize) -> Result<(), String> {
        let len = *self.beam.len.get(from).ok_or("Invalid source beam row")?;
        let dest = self
            .beam
            .len
            .get_mut(to)
            .ok_or("Invalid destination beam row")?;
        if from != to {
            let width = self.m.width * self.suf;
            for l in 0..self.m.layers {
                for i in [cache(l) + 2, cache(l) + 3] {
                    self.beam.compiled.write_input_f32(i, |buf| {
                        buf.copy_within(from * width..(from + 1) * width, to * width);
                    })?;
                }
            }
            *dest = len;
        }
        Ok(())
    }

    fn put_context(
        &self,
        layer: usize,
        pos: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), String> {
        self.beam.put_context(&self.m, layer, pos, key, value)?;
        self.feed.put_context(&self.m, layer, pos, key, value)
    }

    /// Moves the selected suffix into shared context.
    pub fn commit(&mut self, r: usize) -> Result<bool, String> {
        let len = *self.beam.len.get(r).ok_or("Invalid beam row")?;
        if len > self.room() {
            return Ok(false);
        }
        let width = self.m.width * self.suf;
        for l in 0..self.m.layers {
            let ks = self
                .beam
                .compiled
                .read_input_f32(cache(l) + 2, |buf| row(buf, r, width).map(<[_]>::to_vec))??;
            let vs = self
                .beam
                .compiled
                .read_input_f32(cache(l) + 3, |buf| row(buf, r, width).map(<[_]>::to_vec))??;
            for pos in 0..len {
                let key: Vec<_> = ks.iter().skip(pos).step_by(self.suf).copied().collect();
                let mut value = Vec::with_capacity(self.m.width);
                for head in vs.chunks_exact(self.suf * DH) {
                    value.extend_from_slice(row(head, pos, DH)?);
                }
                self.put_context(l, self.len + pos, &key, &value)?;
            }
        }
        self.clear(r)?;
        self.len += len;
        Ok(true)
    }

    pub fn feed(&mut self, token: u16) -> Result<Vec<f32>, String> {
        if self.room() == 0 {
            return Err("Decoder context is full".into());
        }
        let run = self.feed.run(&self.m, self.len, &[(0, token)])?;
        for (l, update) in run.updates.iter().enumerate() {
            self.put_context(
                l,
                self.len,
                update.key.first().ok_or("Missing decoder key")?,
                update.value.first().ok_or("Missing decoder value")?,
            )?;
        }
        self.len += 1;
        self.steps += 1;
        self.secs += run.seconds;
        run.probabilities
            .into_iter()
            .next()
            .ok_or_else(|| "Missing decoder probabilities".into())
    }

    pub fn step(&mut self, rows: &[(usize, u16)]) -> Result<Vec<Vec<f32>>, String> {
        let mut visited = vec![false; self.beam.len.len()];
        for &(r, _) in rows {
            let seen = visited.get_mut(r).ok_or("Invalid beam row")?;
            if std::mem::replace(seen, true) {
                return Err("Duplicate beam row".into());
            }
        }
        let run = self.beam.run(&self.m, self.len, rows)?;
        let (h, s) = (self.m.heads, self.suf);
        for (l, update) in run.updates.iter().enumerate() {
            self.beam.compiled.write_input_f32(cache(l) + 2, |buf| {
                for (&(r, _), key) in rows.iter().zip(&update.key) {
                    let len = *self.beam.len.get(r).ok_or("Invalid beam row")?;
                    if len < s {
                        let dest = row_mut(buf, r, h * DH * s)?;
                        for (cell, &k) in dest.iter_mut().skip(len).step_by(s).zip(key) {
                            *cell = k;
                        }
                    }
                }
                Ok::<_, String>(())
            })??;
            self.beam.compiled.write_input_f32(cache(l) + 3, |buf| {
                for (&(r, _), value) in rows.iter().zip(&update.value) {
                    let len = *self.beam.len.get(r).ok_or("Invalid beam row")?;
                    if len < s {
                        let dest = row_mut(buf, r, h * DH * s)?;
                        for (head, v) in dest.chunks_exact_mut(s * DH).zip(value.chunks_exact(DH)) {
                            row_mut(head, len, DH)?.copy_from_slice(v);
                        }
                    }
                }
                Ok::<_, String>(())
            })??;
        }
        for &(r, _) in rows {
            let len = self.beam.len.get_mut(r).ok_or("Invalid beam row")?;
            *len = (*len + 1).min(s);
        }
        self.steps += 1;
        self.secs += run.seconds;
        Ok(run.probabilities)
    }
}
