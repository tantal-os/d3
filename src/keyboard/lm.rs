use npusvc_sdk::lrt::Compiled;
use std::rc::Rc;

const HEAD_WIDTH: usize = 64;
const SHAPE: &str = "invalid keyboard model shape";
const ROW: &str = "invalid cache row";

pub(super) struct Lm {
    compiled: Compiled,
    context: usize,
    vocabulary: usize,
    width: usize,
    heads: usize,
    layers: usize,
    embeddings: Rc<[f32]>,
    lengths: Vec<usize>,
    frequencies: Vec<f32>,
}

fn product(dims: &[usize]) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |p, d| {
        p.checked_mul(*d).ok_or_else(|| SHAPE.to_owned())
    })
}

fn portion<T>(data: &[T], offset: usize, count: usize) -> Result<&[T], String> {
    let end = offset.checked_add(count).ok_or(SHAPE)?;
    data.get(offset..end).ok_or_else(|| SHAPE.to_owned())
}

fn put<T: Copy>(data: &mut [T], offset: usize, values: &[T]) -> Result<(), String> {
    let end = offset.checked_add(values.len()).ok_or(SHAPE)?;
    data.get_mut(offset..end)
        .ok_or(SHAPE)?
        .copy_from_slice(values);
    Ok(())
}

fn floats(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let (words, trailing) = bytes.as_chunks::<4>();
    if !trailing.is_empty() {
        return Err(SHAPE.into());
    }
    Ok(words.iter().map(|word| f32::from_le_bytes(*word)).collect())
}

impl Lm {
    pub(super) fn new(
        compiled: Compiled,
        embeddings: Rc<[f32]>,
        vocabulary: usize,
        layers: usize,
        context: usize,
        batch: usize,
    ) -> Result<Self, String> {
        if vocabulary < 2
            || !embeddings.len().is_multiple_of(vocabulary)
            || layers == 0
            || layers > 256
            || context == 0
            || batch == 0
        {
            return Err(SHAPE.into());
        }
        let width = embeddings.len() / vocabulary;
        let heads = width / HEAD_WIDTH;
        if heads == 0
            || !width.is_multiple_of(HEAD_WIDTH)
            || compiled.input_count() != 4 + 2 * layers
            || compiled.output_count() != 1 + 2 * layers
        {
            return Err(SHAPE.into());
        }
        let row_bytes = product(&[batch, width, 4])?;
        let mask_bytes = product(&[batch, heads, context.checked_add(1).ok_or(SHAPE)?, 4])?;
        let cache_bytes = product(&[row_bytes, context])?;
        let input_sizes = [row_bytes, row_bytes, row_bytes, mask_bytes]
            .into_iter()
            .chain(std::iter::repeat_n(cache_bytes, 2 * layers));
        for (i, expected) in input_sizes.enumerate() {
            if compiled.input_len(i)? != expected {
                return Err(SHAPE.into());
            }
            compiled.write_input_f32(i, |data| data.fill(0.0))?;
        }
        let frequencies = (0..HEAD_WIDTH / 2)
            .map(|i| 10000f32.powf(-2.0 * i as f32 / HEAD_WIDTH as f32))
            .collect();
        Ok(Self {
            compiled,
            context,
            vocabulary,
            width,
            heads,
            layers,
            embeddings,
            lengths: vec![0; batch],
            frequencies,
        })
    }

    pub(super) fn reset_root(&mut self) -> Result<(), String> {
        *self.lengths.first_mut().ok_or(ROW)? = 0;
        Ok(())
    }

    pub(super) fn copy(&mut self, from: usize, to: usize) -> Result<(), String> {
        let len = *self.lengths.get(from).ok_or(ROW)?;
        if to >= self.lengths.len() {
            return Err(ROW.into());
        }
        if from == to {
            return Ok(());
        }
        for layer in 0..self.layers {
            for value in [false, true] {
                let index = 4 + layer * 2 + usize::from(value);
                let (blocks, stride, used) = self.cache_layout(value, len);
                self.compiled.write_input_f32(index, |data| {
                    for block in 0..blocks {
                        let src = (from * blocks + block) * stride;
                        let dst = (to * blocks + block) * stride;
                        if src + used > data.len() || dst + used > data.len() {
                            return Err(SHAPE.to_owned());
                        }
                        data.copy_within(src..src + used, dst);
                    }
                    Ok(())
                })??;
            }
        }
        *self.lengths.get_mut(to).ok_or(ROW)? = len;
        Ok(())
    }

    fn cache_layout(&self, value: bool, len: usize) -> (usize, usize, usize) {
        if value {
            (self.heads, self.context * HEAD_WIDTH, len * HEAD_WIDTH)
        } else {
            (self.width, self.context, len)
        }
    }

    pub(super) fn import_root(&mut self, root: &Self) -> Result<(), String> {
        let len = *root.lengths.first().ok_or(ROW)?;
        if self.context != root.context || self.width != root.width || self.layers != root.layers {
            return Err(SHAPE.into());
        }
        for layer in 0..self.layers {
            for value in [false, true] {
                let index = 4 + layer * 2 + usize::from(value);
                let (blocks, stride, used) = self.cache_layout(value, len);
                root.compiled.read_input_f32(index, |source| {
                    self.compiled.write_input_f32(index, |target| {
                        for block in 0..blocks {
                            let offset = block * stride;
                            put(target, offset, portion(source, offset, used)?)?;
                        }
                        Ok::<_, String>(())
                    })?
                })??;
            }
        }
        *self.lengths.first_mut().ok_or(ROW)? = len;
        Ok(())
    }

    pub(super) fn step(&mut self, rows: &[(usize, u16)]) -> Result<Vec<Vec<f32>>, String> {
        let Self {
            lengths,
            compiled,
            width,
            heads,
            frequencies,
            embeddings,
            context,
            vocabulary,
            ..
        } = self;
        let (width, heads, context, vocabulary) = (*width, *heads, *context, *vocabulary);
        for &(row, token) in rows {
            if lengths.get(row).is_none_or(|n| *n >= context) || usize::from(token) >= vocabulary {
                return Err(ROW.into());
            }
        }
        compiled.write_input_f32(0, |data| {
            for &(row, token) in rows {
                put(
                    data,
                    row * width,
                    portion(embeddings, usize::from(token) * width, width)?,
                )?;
            }
            Ok::<_, String>(())
        })??;
        for (index, sine) in [(1, false), (2, true)] {
            compiled.write_input_f32(index, |data| {
                for (row, &position) in lengths.iter().enumerate() {
                    for (pair, &frequency) in frequencies.iter().enumerate() {
                        let angle = position as f32 * frequency;
                        let value = if sine { angle.sin() } else { angle.cos() };
                        for head in 0..heads {
                            put(
                                data,
                                (row * heads + head) * HEAD_WIDTH + pair * 2,
                                &[value, value],
                            )?;
                        }
                    }
                }
                Ok::<_, String>(())
            })??;
        }
        compiled.write_input_f32(3, |data| {
            for (row, mask) in data.chunks_exact_mut(context + 1).enumerate() {
                let len = *lengths.get(row / heads).ok_or(ROW)?;
                for (i, value) in mask.iter_mut().enumerate() {
                    *value = if i < len || i == context { 0.0 } else { -1e9 };
                }
            }
            Ok::<_, String>(())
        })??;
        compiled.run()?;
        let mut outputs = compiled.outputs()?.into_iter();
        let logits = floats(&outputs.next().ok_or(SHAPE)?)?;
        if logits.len() != product(&[lengths.len(), vocabulary])? {
            return Err(SHAPE.into());
        }
        for layer in 0..self.layers {
            for value in [false, true] {
                let values = floats(&outputs.next().ok_or(SHAPE)?)?;
                if values.len() != lengths.len() * width {
                    return Err(SHAPE.into());
                }
                compiled.write_input_f32(4 + layer * 2 + usize::from(value), |cache| {
                    for &(row, _) in rows {
                        let pos = *lengths.get(row).ok_or(ROW)?;
                        for head in 0..heads {
                            let block = row * heads + head;
                            let source = portion(&values, block * HEAD_WIDTH, HEAD_WIDTH)?;
                            if value {
                                put(cache, (block * context + pos) * HEAD_WIDTH, source)?;
                            } else {
                                for (i, &v) in source.iter().enumerate() {
                                    *cache
                                        .get_mut((block * HEAD_WIDTH + i) * context + pos)
                                        .ok_or(SHAPE)? = v;
                                }
                            }
                        }
                    }
                    Ok::<_, String>(())
                })??;
            }
        }
        rows.iter()
            .map(|&(row, _)| {
                *lengths.get_mut(row).ok_or(ROW)? += 1;
                let z = portion(&logits, row * vocabulary, vocabulary)?;
                let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let norm = z.iter().map(|x| (x - max).exp()).sum::<f32>().ln() + max;
                if !norm.is_finite() {
                    return Err("non-finite model scores".into());
                }
                Ok(z.iter().map(|x| x - norm).collect())
            })
            .collect()
    }
}
