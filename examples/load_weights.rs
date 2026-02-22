//! Test weight loading — downloads and loads all model components.
//!
//! Usage: cargo run --example load_weights --no-default-features

fn main() -> ace_step_rs::Result<()> {
    tracing_subscriber::fmt::init();

    let device = candle_core::Device::Cpu;
    let dtype = candle_core::DType::F32;

    println!("Loading ACE-Step v1.5 pipeline (CPU, F32)...");
    println!("This will download ~8GB of weights on first run.");

    let _pipeline = ace_step_rs::pipeline::AceStepPipeline::load(&device, dtype)?;

    println!("All weights loaded successfully!");
    Ok(())
}
