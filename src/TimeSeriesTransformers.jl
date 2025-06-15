module TimeSeriesTransformers

import Flux
import Flux: Dense, Dropout, LayerNorm, softmax, relu
using LinearAlgebra
using Random
using Statistics
using NNlib
using Plots
using Base.Threads
using DifferentialEquations

# Export transformer architecture components
export MultiHeadAttention, PositionalEncoding, FeedForward
export TransformerEncoderLayer, ContinuousTransformerModel

# Export utility functions from transformer.jl
export create_causal_mask, set_threading

# Export training functions
export train_continuous_transformer!

# Export delay embedding functions
export DelayEmbeddingProcessor, create_delay_embedding
export get_training_sequences, normalize_value, denormalize_value
export denormalize_predictions, get_normalization_info, get_embedding_data

# Export prediction utilities
export predict_next_values, generate_ensemble_predictions
export analyze_prediction_horizon_scaling

# Export Lorenz system functions
export lorenz63!, generate_lorenz63_data

# Export analysis functions
export combined_prediction_analysis, autocorr

# Include files in dependency order
include("transformer.jl")
include("delay_embedding_utils.jl")
include("lorenz_data.jl")
include("training.jl")
include("prediction_utils.jl")
include("callback.jl")

end
