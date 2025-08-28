## Continuous Transformers for Time Series (Julia)

This repository implements a transformer architecture trained on continuous-valued time series for next-step prediction. It is one of two companion repositories:

- This repo: continuous-input transformer using delay-embedded real-valued sequences.
- Companion repo: discrete-input transformer trained on clustered (discretized) sequences.

If you are looking for the discrete/clustered variant, please refer to the companion repository. This repository is specifically for the continuous-input model.

## Overview

The continuous transformer predicts future values directly from continuous data using delay embeddings. It is built on Flux.jl and includes utilities for data generation (Lorenz-63), delay embedding, training, forecasting, and analysis of prediction quality across horizons.

## Features

- **Continuous input pipeline**: delay-embedding of real-valued time series with optional normalization
- **Transformer encoder**: multi-head attention, feed-forward blocks, layer norm, dropout
- **Continuous output**: direct regression to next value(s) (no discretization)
- **Training utilities**: minibatch training with early stopping and validation batching
- **Prediction & analysis**: single/ensemble forecasts, horizon-scaling RMSE, PDF and ACF comparison plots

## Project Structure

```
ContinuousTransformers.jl
├── src/
│   ├── TimeSeriesTransformers.jl     # Package entry point and exports
│   ├── transformer.jl                # Core transformer components and model (continuous)
│   ├── delay_embedding_utils.jl      # Delay embedding, normalization, data utilities
│   ├── lorenz_data.jl                # Lorenz-63 data generation helpers
│   ├── training.jl                   # Training loop for ContinuousTransformerModel
│   ├── prediction_utils.jl           # Forecasting and analysis helpers
│   └── callback.jl                   # Combined analysis (plots, RMSE scaling, PDFs, ACFs)
├── examples/
│   └── lorenz_transformer_example.jl # End-to-end training + analysis on Lorenz-63
└── test/                             # Test suite (may include legacy discrete tests)
```

## Quickstart

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using TimeSeriesTransformers

# Generate Lorenz-63 data and extract a single component (e.g., y)
data, dt = generate_lorenz63_data(100_000, tspan=(0.0, 5000.0), return_dt=true)
y = data[:, 2]

# Build delay embedding processor (continuous inputs)
embedding_dim = 8
processor = DelayEmbeddingProcessor(y, embedding_dim; normalize=true)

# Define and create the continuous transformer
model = ContinuousTransformerModel(
    input_dim = embedding_dim,
    output_dim = 1,
    d_model = 32,
    num_heads = 8,
    num_layers = 1,
    dropout_rate = 0.1f0,
)

# Train
model, train_losses, val_losses = train_continuous_transformer!(
    model, processor;
    epochs = 60,
    seq_len = 32,
    val_seq_len = 256,
    learning_rate = 1f-3,
    early_stopping_patience = 50,
    verbose = true,
    n_training_steps_per_epoch = 500,
    training_batch_size = 10,
)

# Prepare validation data and run analysis
inputs, targets = get_embedding_data(processor)
n_total = size(inputs, 2)
n_train = Int(round(0.8 * n_total))
val_inputs_full = inputs[:, n_train+1:n_total]
val_targets_full = targets[:, n_train+1:n_total]

combined_plot, preds, obs, horizons, rmse_values = combined_prediction_analysis(
    model, val_inputs_full, val_targets_full;
    n_ens=100, seq_len=32, n_preds_example=100, max_n_preds=150, n_pred_steps=15, seed=42, dt=dt
)
```

See `examples/lorenz_transformer_example.jl` for a larger, fully reproducible script.

## API Summary

From `TimeSeriesTransformers`:

- **Model and layers**: `MultiHeadAttention`, `PositionalEncoding`, `FeedForward`, `TransformerEncoderLayer`, `ContinuousTransformerModel`
- **Threading and masking**: `set_threading`, `create_causal_mask`
- **Delay embedding & normalization**: `DelayEmbeddingProcessor`, `create_delay_embedding`, `get_training_sequences`, `get_embedding_data`, `normalize_value`, `denormalize_value`, `denormalize_predictions`, `get_normalization_info`
- **Training**: `train_continuous_transformer!`
- **Prediction & analysis**: `predict_next_values`, `generate_ensemble_predictions`, `analyze_prediction_horizon_scaling`, `combined_prediction_analysis`, `autocorr`
- **Data generation**: `lorenz63!`, `generate_lorenz63_data`

## Requirements

- Julia ≥ 1.11 (per `Project.toml`)
- Major dependencies: Flux.jl, DifferentialEquations.jl, Plots.jl, Statistics, StatsBase, KernelDensity, Zygote, NNlib

Install all dependencies via:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Notes on the Companion (Discrete) Repository

The discrete repository discretizes continuous series into cluster indices and trains a transformer on sequences of symbols. This repository does not perform discretization and instead works directly on continuous delay-embedded vectors. Use this repo when you want numeric next-value prediction rather than next-cluster classification.

## License

This project is available under the MIT License. See `LICENSE` for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{continuous_transformers_julia,
  title   = {Continuous Transformers for Time Series in Julia},
  author  = {Ludovico Theo Giorgini},
  year    = {2024},
  url     = {<repository-url>}
}
```

## Author

**Ludovico Theo Giorgini**  
Email: ludogio@mit.edu  
Massachusetts Institute of Technology
