# Transformer-Based Time Series Prediction

A Julia implementation of transformer models for time series forecasting with comprehensive analysis capabilities.

## Overview

This project implements transformer neural networks for predicting time series data using a cluster-based approach. The model discretizes continuous time series into clusters and learns to predict future cluster transitions, enabling both short-term and long-term forecasting.

## Features

- **Transformer Architecture**: Custom transformer implementation with attention mechanisms
- **Cluster-Based Prediction**: Discretizes continuous data into clusters for robust prediction
- **Ensemble Forecasting**: Generates probabilistic predictions with confidence intervals
- **Comprehensive Analysis**: Multi-panel analysis including ensemble predictions, delayed forecasts, and statistical comparisons
- **Reproducible Results**: Fixed random seeds ensure consistent analysis across runs
- **Validation Metrics**: Automated accuracy computation and model evaluation

## Project Structure

```
TimeSeriesTransformers/
├── src/
│   ├── analysis_callback.jl    # Comprehensive model analysis and visualization
│   ├── ensemble_predict.jl     # Ensemble prediction utilities
│   ├── transformer_model.jl    # Transformer architecture implementation
│   ├── data_processing.jl      # Time series preprocessing and clustering
│   └── validation.jl           # Model validation and accuracy computation
├── figures/                    # Generated analysis plots
├── data/                      # Input time series data
└── README.md                  # This file
```

## Key Components

### Time Series Processing
- Automatic clustering of continuous data into discrete states
- Sequence generation for supervised learning
- Data normalization and preprocessing

### Transformer Model
- Multi-head attention mechanism
- Positional encoding for temporal relationships
- Configurable architecture (layers, heads, dimensions)
- Softmax output for cluster probability prediction

### Analysis Framework
- **Ensemble Predictions**: 50-member ensembles with confidence intervals
- **Delayed Forecasting**: Multi-step ahead prediction analysis (t+5, t+10, t+20)
- **Statistical Validation**: PDF and autocorrelation function comparisons
- **Reproducible Analysis**: Fixed starting points and random seeds

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TimeSeriesTransformers
```

2. Install required Julia packages:
```julia
using Pkg
Pkg.add(["Flux", "Plots", "Statistics", "StatsBase", "KernelDensity", "Random"])
```

## Usage

### Basic Model Training
```julia
include("src/transformer_model.jl")
include("src/data_processing.jl")

# Load and preprocess data
processor = TimeSeriesProcessor(data, n_clusters=10)
X_train, y_train = create_sequences(processor, sequence_length=20)

# Train transformer model
model = create_transformer_model(vocab_size=10, d_model=64, n_heads=4, n_layers=2)
train!(model, X_train, y_train, epochs=100)
```

### Comprehensive Analysis
```julia
include("src/analysis_callback.jl")

# Initialize analysis parameters for reproducibility
initialize_analysis_params!(X_val, y_val)

# Generate comprehensive analysis plot
fig = comprehensive_analysis_callback(
    model, processor, X_val, y_val, y_data,
    save_path="figures/model_analysis.png",
    verbose=true
)
```

### Ensemble Predictions
```julia
include("src/ensemble_predict.jl")

# Generate ensemble forecasts
trajectories, values = ensemble_predict(
    model, processor, initial_sequence, 
    n_ensemble=50, pred_steps=20
)

# Calculate prediction statistics
ensemble_mean = mean(values, dims=2)
confidence_intervals = [quantile(values[i, :], [0.05, 0.95]) for i in 1:pred_steps]
```

## Analysis Output

The comprehensive analysis generates a 6-row visualization:

1. **Rows 1-2**: Four ensemble predictions from different validation points with confidence intervals
2. **Rows 3-5**: Delayed prediction comparisons at t+5, t+10, and t+20 steps
3. **Row 6**: Statistical analysis comparing predicted vs. true probability distributions and autocorrelations

Each analysis uses fixed starting points and random seeds to ensure reproducible results across training epochs.

## Model Validation

The framework includes automated validation metrics:
- **Classification Accuracy**: Percentage of correctly predicted cluster transitions
- **Mean Absolute Error**: Average prediction error for delayed forecasts
- **Statistical Divergence**: Comparison of predicted vs. true data distributions
- **Temporal Correlation**: Autocorrelation function analysis

## Key Features for Research

### Reproducibility
- Fixed random seeds for all stochastic components
- Deterministic starting points for analysis
- Consistent ensemble generation across runs

### Comprehensive Evaluation
- Multiple prediction horizons (1-step to 20-step ahead)
- Probabilistic and deterministic forecasting modes
- Statistical validation of long-term behavior

### Visualization
- Publication-ready plots with confidence intervals
- Multi-panel analysis for complete model assessment
- Automatic figure generation and saving

## Dependencies

- **Julia 1.6+**
- **Flux.jl**: Neural network framework
- **Plots.jl**: Visualization and plotting
- **Statistics.jl**: Statistical functions
- **StatsBase.jl**: Extended statistical functions
- **KernelDensity.jl**: Probability density estimation
- **Random.jl**: Random number generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request with detailed description

## License

This project is available under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{transformer_timeseries,
  title={Transformer-Based Time Series Prediction with Comprehensive Analysis},
  author={Ludovico Theo Giorgini},
  year={2024},
  url={[Repository URL]}
}
```

## Author

**Ludovico Theo Giorgini**  
Email: ludogio@mit.edu  
Massachusetts Institute of Technology

## Contact

For questions or issues, please open a GitHub issue or contact ludogio@mit.edu.
