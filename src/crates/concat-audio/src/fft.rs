//! 基数 2 快速傅里叶变换（FFT）。

use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};

/// 单精度复数。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    /// 实部。
    pub re: f32,
    /// 虚部。
    pub im: f32,
}

impl Complex {
    /// 由实部与虚部构造复数。
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// 模长 `|z|`。
    pub fn norm(self) -> f32 {
        self.re.hypot(self.im)
    }

    /// 乘以实数标量。
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.re * s, self.im * s)
    }

    /// 共轭复数。
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

/// 原地 FFT。`inverse` 为真时做逆变换（含 `1/n` 归一化）。
///
/// # Panics
///
/// 长度不是 2 的幂时 panic。
pub fn fft_in_place(buf: &mut [Complex], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "FFT 长度必须是 2 的幂: {n}");
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            buf.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let ang = sign * 2.0 * PI / len as f64;
        let tw: Vec<Complex> = (0..half)
            .map(|k| {
                let a = ang * k as f64;
                Complex::new(a.cos() as f32, a.sin() as f32)
            })
            .collect();
        for chunk in buf.chunks_mut(len) {
            let (lo, hi) = chunk.split_at_mut(half);
            for ((u, v), w) in lo.iter_mut().zip(hi.iter_mut()).zip(&tw) {
                let a = *u;
                let b = *v * *w;
                *u = a + b;
                *v = a - b;
            }
        }
        len <<= 1;
    }
    if inverse {
        let s = 1.0 / n as f32;
        for c in buf.iter_mut() {
            *c = c.scale(s);
        }
    }
}

/// 复数正变换，返回新向量。
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    let mut buf = input.to_vec();
    fft_in_place(&mut buf, false);
    buf
}

/// 复数逆变换（含 `1/n` 归一化），返回新向量。
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let mut buf = input.to_vec();
    fft_in_place(&mut buf, true);
    buf
}

/// 实信号正变换，返回前 `n/2+1` 个频点。
pub fn rfft(input: &[f32]) -> Vec<Complex> {
    let mut buf: Vec<Complex> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft_in_place(&mut buf, false);
    buf.truncate(input.len() / 2 + 1);
    buf
}

/// 由 `n/2+1` 个频点（按厄米对称补全）恢复长度为 `n` 的实信号。
///
/// # Panics
///
/// `spec.len() != n/2+1` 或 `n` 不是 2 的幂时 panic。
pub fn irfft(spec: &[Complex], n: usize) -> Vec<f32> {
    assert_eq!(spec.len(), n / 2 + 1, "频点数与 n 不匹配");
    let mut buf = vec![Complex::default(); n];
    buf[..spec.len()].copy_from_slice(spec);
    for (k, slot) in buf.iter_mut().enumerate().skip(n / 2 + 1) {
        *slot = spec[n - k].conj();
    }
    fft_in_place(&mut buf, true);
    buf.into_iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn fft_roundtrip() {
        let x: Vec<Complex> = noise(1024, 1)
            .into_iter()
            .zip(noise(1024, 2))
            .map(|(a, b)| Complex::new(a, b))
            .collect();
        let y = ifft(&fft(&x));
        for (a, b) in x.iter().zip(&y) {
            assert!((*a - *b).norm() < 1e-5);
        }
    }

    #[test]
    fn fft_matches_naive_dft() {
        let x = noise(16, 7);
        let spec = rfft(&x);
        for (k, z) in spec.iter().enumerate() {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (t, &v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * t) as f64 / 16.0;
                re += v as f64 * a.cos();
                im += v as f64 * a.sin();
            }
            assert!((z.re as f64 - re).abs() < 1e-4);
            assert!((z.im as f64 - im).abs() < 1e-4);
        }
    }

    #[test]
    fn rfft_roundtrip() {
        let x = noise(256, 3);
        let y = irfft(&rfft(&x), 256);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
