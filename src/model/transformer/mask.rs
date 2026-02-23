//! Attention mask creation for SDPA mode.
//!
//! Creates 4D additive masks [1, 1, S, S] with 0.0 for visible positions
//! and -inf for masked positions. Supports sliding window (bidirectional).

use candle_core::{DType, Device, Result, Tensor};

/// Create a 4D attention mask [1, 1, seq_len, seq_len].
///
/// - `is_causal`: if true, apply causal mask (lower-triangular)
/// - `sliding_window`: if Some(w), restrict attention to ±w positions
///
/// For ACE-Step v1.5, all attention is bidirectional (is_causal=false).
pub fn create_4d_mask(
    seq_len: usize,
    dtype: DType,
    device: &Device,
    is_causal: bool,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    let min_val = match dtype {
        DType::F32 => f64::from(f32::MIN),
        DType::F16 => f64::from(half::f16::MIN),
        DType::BF16 => f64::from(half::bf16::MIN),
        DType::F64 => f64::MIN,
        _ => f64::from(f32::MIN),
    };

    // Start with all zeros (all visible)
    let mut mask_data = vec![0.0f64; seq_len * seq_len];

    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut masked = false;

            // Causal mask: can only attend to positions <= i
            if is_causal && j > i {
                masked = true;
            }

            // Sliding window: bidirectional local attention
            if let Some(w) = sliding_window {
                let diff = i.abs_diff(j);
                if diff > w {
                    masked = true;
                }
            }

            if masked {
                mask_data[i * seq_len + j] = min_val;
            }
        }
    }

    let mask = Tensor::new(mask_data.as_slice(), device)?
        .reshape((1, 1, seq_len, seq_len))?
        .to_dtype(dtype)?;
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_full_bidirectional_mask() {
        let mask = create_4d_mask(4, DType::F32, &Device::Cpu, false, None).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
        // All zeros — everything visible
        let sum: f32 = mask.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(sum < 1e-6);
    }

    #[test]
    fn test_sliding_window_mask() {
        let mask = create_4d_mask(6, DType::F32, &Device::Cpu, false, Some(1)).unwrap();
        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Position (0,0) should be 0 (visible)
        assert_eq!(vals[0], 0.0);
        // Position (0,2) should be -inf (distance 2 > window 1)
        assert!(vals[2] < -1e30);
        // Position (0,1) should be 0 (distance 1 <= window 1)
        assert_eq!(vals[1], 0.0);
    }
}
