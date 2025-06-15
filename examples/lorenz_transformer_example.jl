using TimeSeriesTransformers
using Flux
using LinearAlgebra
using Random
using Statistics
using NNlib
using Plots
using KernelDensity

# Set seed for reproducibility
Random.seed!(42)

println("=== Lorenz Time Series Transformer Example ===")
println("Julia threads: $(Threads.nthreads())")

# ===================================================================
# 1. Data Generation and Preprocessing
# ===================================================================

println("\n1. Generating Lorenz 63 time series data...")

# Generate chaotic time series from Lorenz 63 system
# The Lorenz system is a classic example of deterministic chaos
data, dt_lorenz = generate_lorenz63_data(100000, tspan=(0.0, 5000.0), return_dt=true)

# Extract the y-variable for time series prediction
y_data = vec(data[:, 2])  # Convert column to vector
println("y_data type: $(typeof(y_data))")
println("y_data size: $(size(y_data))")

println("Generated $(length(y_data)) data points")
println("Data range: [$(round(minimum(y_data), digits=3)), $(round(maximum(y_data), digits=3))]")
println("Time step (dt): $dt_lorenz")

# Create delay embedding instead of discrete clusters
embedding_dim = 8  # Delay embedding dimension (m)
println("\n2. Creating $(embedding_dim)-dimensional delay embedding...")

# Create delay embedding processor with normalization
processor = DelayEmbeddingProcessor(y_data, embedding_dim; normalize=true)
println("Embedding dimension: $(embedding_dim)")
println("Embedded data shape: $(size(processor.embedded_data))")

# Print normalization information
norm_info = get_normalization_info(processor)
println("Normalization info:")
println("  - Enabled: $(norm_info.normalized)")
if norm_info.normalized
    println("  - Original range: [$(round(norm_info.original_range[1], digits=3)), $(round(norm_info.original_range[2], digits=3))]")
    println("  - Normalized range: [$(round(norm_info.normalized_range[1], digits=3)), $(round(norm_info.normalized_range[2], digits=3))]")
    println("  - Mean: $(round(norm_info.mean[1], digits=3))")
    println("  - Std: $(round(norm_info.std[1], digits=3))")
end

# ===================================================================
# 2. Model Configuration
# ===================================================================

println("\n3. Configuring transformer model...")

# Model hyperparameters - removed sequence_length as it's no longer needed
d_model = 32           # Model dimension
num_heads = 8          # Number of attention heads
num_layers = 1         # Number of transformer layers
dropout_rate = 0.1f0   # Dropout rate for regularization (Float32)

# Neural ODE parameters
T = 1.0f0             # Integration time for Neural ODE
dt = dt_lorenz        # Time step for Neural ODE integration
node_layers = Int[64, 32] # Internal layers: latent_dim -> 64 -> 32 -> latent_dim

# Validate configuration
head_dim = d_model ÷ num_heads
if d_model % num_heads != 0
    error("d_model ($d_model) must be divisible by num_heads ($num_heads)")
end

println("Model configuration:")
println("  - Model dimension: $d_model")
println("  - Attention heads: $num_heads (head dimension: $head_dim)")
println("  - Transformer layers: $num_layers")
println("  - Dropout rate: $dropout_rate")
println("  - Input dimension: $embedding_dim")
println("  - Output dimension: 1")
println("  - Neural ODE integration time (T): $T")
println("  - Neural ODE time step (dt): $dt")
println("  - Neural ODE internal layers: $node_layers")

# Create the continuous transformer model
model = ContinuousTransformerModel(
    input_dim = embedding_dim,
    output_dim = 1,  # Predicting single next value
    d_model = d_model,
    latent_dim = 3, 
    num_heads = num_heads,
    num_layers = num_layers,
    dropout_rate = dropout_rate,
    node_layers = node_layers,  # Add the internal layers specification
    T = 10*dt,           # Integration time parameter
    dt = dt          # Time step parameter
)

# Count parameters
param_count = sum(length, Flux.params(model))
println("Total trainable parameters: $param_count")

# ===================================================================
# 3. Training and Evaluation
# ===================================================================

println("\n4. Training transformer model...")

model, train_losses, val_losses = train_continuous_transformer!(
    model, 
    processor;
    epochs = 120,
    seq_len = 32,                         # Sequence length for training
    val_seq_len = 256,                    # Sequence length for validation batches (adjust as needed, 256 is the default)
    learning_rate = 1f-3,
    early_stopping_patience = 50,
    verbose = true,
    n_training_steps_per_epoch = 500,     # Number of gradient updates per epoch
    training_batch_size = 10              # Number of sequences per 3D batch for training
)

println("Training completed successfully!")

##
# ===================================================================
# 4. Get validation data for analysis
# ===================================================================

println("\n5. Preparing validation data for analysis...")

# Get the full embedding data
inputs, targets = get_embedding_data(processor)
n_total_samples = size(inputs, 2)
n_train_samples = Int(round(0.8 * n_total_samples))

# Extract validation data (same split as used in training)
val_inputs_full = inputs[:, n_train_samples+1:n_total_samples]
val_targets_full = targets[:, n_train_samples+1:n_total_samples]

println("Validation data prepared: $(size(val_inputs_full, 2)) samples")

# ===================================================================
# 5. Run prediction analysis
# ===================================================================

println("\n6. Running combined prediction analysis...")

# dt_lorenz is now automatically available from data generation
combined_plot, ensemble_preds, ensemble_obs, horizons, rmse_values = combined_prediction_analysis(
    model, val_inputs_full, val_targets_full;
    n_ens=500, seq_len=32, n_preds_example=100, max_n_preds=150, n_pred_steps=15, seed=42,
    dt=dt_lorenz
)

# Save the combined plot
savefig(combined_plot, "lorenz_transformer_analysis.png")
println("Figure saved as 'lorenz_transformer_analysis.png'")
display(combined_plot)