"""
Test suite for TimeSeriesTransformers.jl

This test suite covers the core functionality needed before publishing:
1. Package loading and basic imports
2. Data generation capabilities  
3. Time series processor functionality
4. Model creation and basic operations
5. Training pipeline (minimal test)
6. Prediction and generation utilities
"""

using Test
using TimeSeriesTransformers
using Random
using Statistics
using Flux

# Set seed for reproducible tests
Random.seed!(12345)

println("Starting TimeSeriesTransformers.jl test suite...")

# Include test modules
include("test_data_generation.jl")
include("test_processor.jl") 
include("test_model.jl")
include("test_training.jl")
include("test_prediction.jl")

println("All tests completed successfully!")
