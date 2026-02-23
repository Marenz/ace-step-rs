//! Finite Scalar Quantization (FSQ).
//!
//! Quantizes continuous vectors into discrete codes using per-dimension
//! bounded quantization with fixed levels [8, 8, 8, 5, 5, 5].

use candle_core::{Result, Tensor};

/// FSQ quantizer matching `vector_quantize_pytorch.ResidualFSQ`.
///
/// For inference we only need the forward pass (quantize) and
/// `get_output_from_indices` (decode codes back to continuous).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResidualFsq {
    /// Project from hidden_size to FSQ dim
    project_in: Option<candle_nn::Linear>,
    /// Project from FSQ dim back to hidden_size
    project_out: Option<candle_nn::Linear>,
    /// Number of levels per dimension, e.g. [8, 8, 8, 5, 5, 5]
    levels: Vec<usize>,
    /// Codebook embeddings for each quantizer
    codebook: Tensor,
}

impl ResidualFsq {
    pub fn new(
        _dim: usize,
        _levels: &[usize],
        _num_quantizers: usize,
        _vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        // TODO: implement FSQ loading
        // For turbo inference without cover mode, FSQ is not needed
        // (only needed for tokenize/detokenize in cover song mode)
        todo!("FSQ not yet implemented — only needed for cover mode")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_fsq_placeholder() {
        // FSQ implementation is deferred — not needed for basic text-to-music
    }
}
