//! Hushward: keep the entire convolutional encoder/decoder on the GPU. Original
//! FP32 parameters, padding, normalization and subpixel ordering are preserved.
use super::*;
use burn::tensor::{module::conv2d, ops::ConvOptions, Device, Tensor, TensorData};
type T<const D: usize> = Tensor<D>;

fn tensor<const D: usize>(values: &[f32], shape: [usize; D], device: &Device) -> T<D> {
    Tensor::<D>::from_data(TensorData::new(values.to_vec(), shape), device)
}

struct GConv {
    weight: T<4>,
    bias: T<1>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
}
impl GConv {
    fn new(c: &Conv, device: &Device) -> Self {
        Self {
            weight: tensor(&c.weight, [c.cout, c.cin, c.kt, c.kf], device),
            bias: tensor(&c.bias, [c.cout], device),
            stride: [c.stride.0, c.stride.1],
            padding: [c.pad.0, c.pad.1],
            dilation: [c.dilation, 1],
        }
    }
    fn run(&self, x: T<4>) -> T<4> {
        conv2d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new(self.stride, self.padding, self.dilation, 1),
        )
    }
}
struct Norm {
    weight: T<4>,
    bias: T<4>,
    prelu: T<4>,
}
impl Norm {
    fn new(n: &NormAct, d: &Device) -> Self {
        let c = n.weight.len();
        Self {
            weight: tensor(&n.weight, [1, c, 1, 1], d),
            bias: tensor(&n.bias, [1, c, 1, 1], d),
            prelu: tensor(&n.prelu, [1, c, 1, 1], d),
        }
    }
    fn run(&self, x: T<4>) -> T<4> {
        let mean = x.clone().mean_dim(2).mean_dim(3);
        let centered = x - mean;
        let variance = centered.clone().square().mean_dim(2).mean_dim(3);
        let x = centered / (variance + 1e-5).sqrt() * self.weight.clone() + self.bias.clone();
        x.clone().clamp_min(0.0) + x.clamp_max(0.0) * self.prelu.clone()
    }
}
struct GBlock {
    conv: GConv,
    norm: Norm,
}
impl GBlock {
    fn new(c: &ConvBlock, d: &Device) -> Self {
        Self {
            conv: GConv::new(&c.conv, d),
            norm: Norm::new(&c.act, d),
        }
    }
    fn run(&self, x: T<4>) -> T<4> {
        self.norm.run(self.conv.run(x))
    }
}
struct Dense {
    layers: Vec<GBlock>,
}
impl Dense {
    fn new(dense: &DenseBlock, d: &Device) -> Self {
        Self {
            layers: dense.layers.iter().map(|c| GBlock::new(c, d)).collect(),
        }
    }
    fn run(&self, mut x: T<4>) -> T<4> {
        for (i, layer) in self.layers.iter().enumerate() {
            let y = layer.run(x.clone());
            if i + 1 == self.layers.len() {
                return y;
            }
            x = Tensor::cat(vec![y, x], 1);
        }
        unreachable!()
    }
}
struct GDecoder {
    dense: Dense,
    up1: GBlock,
    up2: GBlock,
    heads: Vec<(T<4>, T<1>)>,
}
fn shuffle_gpu(x: T<4>, along_time: bool) -> T<4> {
    let [b, c, t, f] = x.dims();
    let channels = c / if along_time { 4 } else { 2 };
    let factor = c / channels;
    let x = x.reshape([b, factor, channels, t, f]);
    if along_time {
        x.permute([0, 2, 3, 1, 4])
            .reshape([b, channels, t * factor, f])
    } else {
        x.permute([0, 2, 3, 4, 1])
            .reshape([b, channels, t, f * factor])
    }
}
impl GDecoder {
    fn new(dec: &Decoder, d: &Device) -> Self {
        Self {
            dense: Dense::new(&dec.dense, d),
            up1: GBlock::new(&dec.up1, d),
            up2: GBlock::new(&dec.up2, d),
            heads: dec
                .heads
                .iter()
                .map(|(w, b)| (tensor(w, [1, w.len(), 1, 1], d), tensor(&[*b], [1], d)))
                .collect(),
        }
    }
    fn run(&self, x: T<4>) -> Vec<T<4>> {
        let x = self.dense.run(x);
        let x = self.up1.norm.run(shuffle_gpu(self.up1.conv.run(x), false));
        let x = self.up2.norm.run(shuffle_gpu(self.up2.conv.run(x), true));
        self.heads
            .iter()
            .map(|(w, b)| {
                conv2d(
                    x.clone(),
                    w.clone(),
                    Some(b.clone()),
                    ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
                )
            })
            .collect()
    }
}
pub(super) struct Full {
    encoder_in: GBlock,
    encoder_dense: Dense,
    encoder_out: GBlock,
    mag: GDecoder,
    pha: GDecoder,
    device: Device,
}
impl Full {
    pub(super) fn new(model: &Offline, d: &Device) -> Self {
        Self {
            encoder_in: GBlock::new(&model.encoder_in, d),
            encoder_dense: Dense::new(&model.encoder_dense, d),
            encoder_out: GBlock::new(&model.encoder_out, d),
            mag: GDecoder::new(&model.mag, d),
            pha: GDecoder::new(&model.pha, d),
            device: d.clone(),
        }
    }
    pub(super) fn forward(
        &self,
        mag: &[Vec<f32>],
        pha: &[Vec<f32>],
        mamba: &crate::gpu::Model,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (bins, frames) = (mag.len(), mag[0].len());
        let (time, freq) = (frames + 2, bins + 2);
        let mut data = vec![0.0; 2 * time * freq];
        for (ch, spec) in [mag, pha].iter().enumerate() {
            for (f, row) in spec.iter().enumerate() {
                for (t, v) in row.iter().enumerate() {
                    data[(ch * time + t) * freq + f] = *v;
                }
            }
        }
        let x = Tensor::<4>::from_data(TensorData::new(data, [1, 2, time, freq]), &self.device);
        let x = self
            .encoder_out
            .run(self.encoder_dense.run(self.encoder_in.run(x)));
        let [_, channels, time, freq] = x.dims();
        let rows = x.permute([0, 3, 2, 1]).reshape([time * freq, channels]);
        let x = mamba
            .mamba_tensor(rows, time, freq, channels)
            .reshape([1, freq, time, channels])
            .permute([0, 3, 2, 1]);
        let magnitude = self.mag.run(x.clone()).remove(0);
        let mut phase = self.pha.run(x);
        let imaginary = phase.pop().unwrap();
        let real = phase.pop().unwrap();
        let phase = imaginary.atan2(real);
        let values = Tensor::cat(vec![magnitude, phase], 1)
            .slice([0..1, 0..2, 0..frames, 0..bins])
            .permute([0, 1, 3, 2])
            .reshape([2 * bins, frames])
            .into_data()
            .try_into_vec::<f32>()
            .expect("FP32 model output");
        let mut rows = values.chunks_exact(frames).map(<[f32]>::to_vec);
        let magnitude = rows.by_ref().take(bins).collect();
        (magnitude, rows.collect())
    }
}
