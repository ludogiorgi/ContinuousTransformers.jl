import Flux
import Flux: Dense, Dropout, LayerNorm, softmax, relu
using LinearAlgebra
using Random
using Statistics
using Plots

# Add threading capabilities
using Base.Threads

using DifferentialEquations

"""
    lorenz63!(du, u, p, t)

Lorenz 63 system differential equation.
"""
function lorenz63!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

"""
    generate_lorenz63_data(n_points::Int; tspan=(0.0, 100.0), dt=nothing, σ=10.0, ρ=28.0, β=8/3)

Generate time series data from the Lorenz 63 system.

# Arguments
- `n_points::Int`: Number of data points to generate
- `tspan`: Time span as (start_time, end_time)
- `dt`: Time step (if nothing, automatically calculated)
- `σ`, `ρ`, `β`: Lorenz system parameters

# Returns
- Matrix of size (n_points, 3) containing [x, y, z] coordinates
"""
function generate_lorenz63_data(n_points::Int; tspan=(0.0, 100.0), dt=nothing, σ=10.0, ρ=28.0, β=8/3)
    # Initial conditions
    u0 = [1.0, 1.0, 1.0]
    
    # Parameters
    p = [σ, ρ, β]
    
    # Calculate time step if not provided
    if dt === nothing
        dt = (tspan[2] - tspan[1]) / (n_points - 1)
    end
    
    # Create time vector
    t = range(tspan[1], tspan[2], length=n_points)
    
    # Define the ODE problem
    prob = ODEProblem(lorenz63!, u0, tspan, p)
    
    # Solve the ODE
    sol = solve(prob, Tsit5(), saveat=t, reltol=1e-8, abstol=1e-8)
    
    # Convert to matrix format
    data = hcat(sol.u...)'  # Transpose to get (n_points, 3)
    
    return Float32.(data)
end

"""
    generate_lorenz63_data(n_points::Int, tspan::Tuple)

Convenience method for generate_lorenz63_data with positional tspan argument.
"""
function generate_lorenz63_data(n_points::Int, tspan::Tuple)
    return generate_lorenz63_data(n_points; tspan=tspan)
end

# =====================================================

"""
    DelayEmbeddingProcessor

Processor for creating delay embeddings from time series data with optional normalization.
"""

using Statistics

struct DelayEmbeddingProcessor
    original_data::Vector{Float32}
    normalized_data::Vector{Float32}
    embedding_dim::Int
    embedded_data::Matrix{Float32}
    # Normalization parameters
    data_mean::Float32
    data_std::Float32
    normalize::Bool
    
    function DelayEmbeddingProcessor(data::Vector{<:Real}, embedding_dim::Int; normalize::Bool=true)
        data_f32 = Float32.(data)
        
        # Compute normalization parameters
        if normalize
            data_mean = Float32(Statistics.mean(data_f32))
            data_std = Float32(Statistics.std(data_f32))
            # Avoid division by zero
            if data_std < 1e-8
                @warn "Data has very small standard deviation ($data_std), using std=1.0"
                data_std = 1.0f0
            end
            normalized_data = (data_f32 .- data_mean) ./ data_std
        else
            data_mean = 0.0f0
            data_std = 1.0f0
            normalized_data = copy(data_f32)
        end
        
        # Create delay embedding from normalized data
        embedded = create_delay_embedding(normalized_data, embedding_dim)
        new(data_f32, normalized_data, embedding_dim, embedded, data_mean, data_std, normalize)
    end
end

"""
    create_delay_embedding(data, embedding_dim)

Create delay embedding matrix from 1D time series.
Returns matrix of size (length(data) - embedding_dim + 1, embedding_dim)
where each row is [y(t), y(t-1), ..., y(t-embedding_dim+1)]
"""
function create_delay_embedding(data::Vector{Float32}, embedding_dim::Int)
    n = length(data)
    if n < embedding_dim
        error("Data length ($n) must be at least embedding dimension ($embedding_dim)")
    end
    
    embedded_length = n - embedding_dim + 1
    embedded = Matrix{Float32}(undef, embedded_length, embedding_dim)
    
    for i in 1:embedded_length
        for j in 1:embedding_dim
            embedded[i, j] = data[i + j - 1]
        end
    end
    
    return embedded
end

"""
    get_training_sequences(processor, sequence_length)

Generate training sequences from delay-embedded data.
Returns (inputs, targets) where inputs are sequences of embedded vectors
and targets are the next values in the normalized time series.
"""
function get_training_sequences(processor::DelayEmbeddingProcessor, sequence_length::Int)
    embedded_data = processor.embedded_data
    normalized_data = processor.normalized_data  # Use normalized data for targets
    embedding_dim = processor.embedding_dim
    
    n_embedded = size(embedded_data, 1)
    n_sequences = n_embedded - sequence_length
    
    if n_sequences <= 0
        error("Not enough data for sequences of length $sequence_length")
    end
    
    # Input sequences: (sequence_length, embedding_dim, n_sequences)
    inputs = Array{Float32}(undef, sequence_length, embedding_dim, n_sequences)
    # Target values: (1, n_sequences) - predicting next value
    targets = Array{Float32}(undef, 1, n_sequences)
    
    for i in 1:n_sequences
        # Input sequence from embedded data - fix the dimension assignment
        for t in 1:sequence_length
            for d in 1:embedding_dim
                inputs[t, d, i] = embedded_data[i + t - 1, d]
            end
        end
        
        # Target is the next value in normalized time series
        # The target corresponds to the next point after the last embedding vector
        target_idx = i + sequence_length + embedding_dim - 1
        if target_idx <= length(normalized_data)
            targets[1, i] = normalized_data[target_idx]
        else
            targets[1, i] = normalized_data[end]  # Use last available value
        end
    end
    
    return inputs, targets
end

"""
    get_embedding_data(processor)

Generate input-target pairs from delay embedding for direct prediction.
Returns (inputs, targets) where:
- inputs: (embedding_dim, n_samples) - each column is a delay embedding vector [y_i, y_{i+1}, ..., y_{i+embedding_dim-1}]  
- targets: (1, n_samples) - each target is the next value y_{i+embedding_dim}
"""
function get_embedding_data(processor::DelayEmbeddingProcessor)
    normalized_data = processor.normalized_data
    embedding_dim = processor.embedding_dim
    
    n_samples = length(normalized_data) - embedding_dim
    if n_samples <= 0
        error("Not enough data for embedding dimension $embedding_dim")
    end
    
    # Input matrix: (embedding_dim, n_samples)
    inputs = Array{Float32}(undef, embedding_dim, n_samples)
    # Target matrix: (1, n_samples)
    targets = Array{Float32}(undef, 1, n_samples)
    
    for i in 1:n_samples
        # Input: [y_i, y_{i+1}, ..., y_{i+embedding_dim-1}]
        inputs[:, i] = normalized_data[i:i+embedding_dim-1]
        # Target: y_{i+embedding_dim} (the next value after the embedding window)
        targets[1, i] = normalized_data[i+embedding_dim]
    end
    
    return inputs, targets
end

"""
    normalize_value(processor, value)

Normalize a single value using the processor's normalization parameters.
"""
function normalize_value(processor::DelayEmbeddingProcessor, value::Real)
    if !processor.normalize
        return Float32(value)
    end
    return Float32((value - processor.data_mean) / processor.data_std)
end

"""
    denormalize_value(processor, normalized_value)

Convert a normalized value back to original scale.
"""
function denormalize_value(processor::DelayEmbeddingProcessor, normalized_value::Real)
    if !processor.normalize
        return Float32(normalized_value)
    end
    return Float32(normalized_value * processor.data_std + processor.data_mean)
end

"""
    denormalize_predictions(processor, predictions)

Convert normalized predictions back to original scale.
"""
function denormalize_predictions(processor::DelayEmbeddingProcessor, predictions::Vector{<:Real})
    if !processor.normalize
        return Float32.(predictions)
    end
    return Float32.([denormalize_value(processor, pred) for pred in predictions])
end

"""
    get_normalization_info(processor)

Get normalization parameters for the processor.
"""
function get_normalization_info(processor::DelayEmbeddingProcessor)
    return (
        normalized=processor.normalize,
        mean=processor.data_mean,
        std=processor.data_std,
        original_range=(minimum(processor.original_data), maximum(processor.original_data)),
        normalized_range=processor.normalize ? (minimum(processor.normalized_data), maximum(processor.normalized_data)) : nothing
    )
end

# =====================================================

# Cache for storing precomputed causal masks
const CAUSAL_MASK_CACHE = Dict{Int, Matrix{Float32}}()

# Global flag to control threading (disabled during gradient computation)
const USE_THREADING = Ref(true)

"""
    set_threading(enabled::Bool)

Enable or disable threading in the transformer. Threading should be disabled
during gradient computation with Zygote.
"""
function set_threading(enabled::Bool)
    USE_THREADING[] = enabled
end

"""
    create_causal_mask(seq_len)

Create a causal attention mask that prevents attending to future positions.
Uses a cache for improved performance.
"""
function create_causal_mask(seq_len)
    # Check if mask is already cached
    if haskey(CAUSAL_MASK_CACHE, seq_len)
        return CAUSAL_MASK_CACHE[seq_len]
    end
    
    # Create row indices and column indices for broadcasting
    row_indices = reshape(1:seq_len, seq_len, 1)
    col_indices = reshape(1:seq_len, 1, seq_len)
    
    # Create the mask directly using broadcasting
    mask = ifelse.(row_indices .>= col_indices, 0.0f0, Float32(-Inf))
    
    # Cache the mask for future use
    CAUSAL_MASK_CACHE[seq_len] = mask
    return mask
end

"""
    MultiHeadAttention

Multi-head self-attention layer as described in the transformer architecture.
Modified to support num_heads > d_model with explicit head dimension.
"""
struct MultiHeadAttention
    num_heads::Int
    head_dim::Int
    W_q::Dense
    W_k::Dense
    W_v::Dense
    W_o::Dense
    dropout::Dropout
    scale::Float32
    
    function MultiHeadAttention(d_model::Int, num_heads::Int, head_dim::Int=0, dropout_rate::Float64=0.1)
        # If head_dim is not provided or is 0, calculate it from d_model (backward compatibility)
        actual_head_dim = head_dim <= 0 ? div(d_model, num_heads) : head_dim
        
        # No longer require d_model to be divisible by num_heads
        if head_dim <= 0 && d_model % num_heads != 0
            @warn "d_model ($d_model) is not divisible by num_heads ($num_heads). Using head_dim = $(actual_head_dim)"
        end
        
        scale = Float32(1 / sqrt(actual_head_dim))
        
        # Project from d_model to (num_heads * head_dim) for queries, keys, and values
        total_dim = num_heads * actual_head_dim
        W_q = Dense(d_model, total_dim)
        W_k = Dense(d_model, total_dim)
        W_v = Dense(d_model, total_dim)
        
        # Project back from (num_heads * head_dim) to d_model
        W_o = Dense(total_dim, d_model)
        
        dropout = Dropout(dropout_rate)
        
        new(num_heads, actual_head_dim, W_q, W_k, W_v, W_o, dropout, scale)
    end
    
    # Constructor for functor reconstruction (used by Flux optimizer)
    MultiHeadAttention(num_heads, head_dim, W_q, W_k, W_v, W_o, dropout, scale) = 
        new(num_heads, head_dim, W_q, W_k, W_v, W_o, dropout, scale)
end

# Add Flux functor declaration for MultiHeadAttention
Flux.@functor MultiHeadAttention

"""
    (mha::MultiHeadAttention)(query::AbstractArray; mask=nothing)

Self-attention case (query = key = value).
"""
function (mha::MultiHeadAttention)(query::AbstractArray; mask=nothing)
    return mha(query, query, query; mask=mask)
end

"""
    (mha::MultiHeadAttention)(query::AbstractArray, key::AbstractArray; mask=nothing)

Key-value attention with same key and value.
"""
function (mha::MultiHeadAttention)(query::AbstractArray, key::AbstractArray; mask=nothing)
    return mha(query, key, key; mask=mask)
end

"""
    (mha::MultiHeadAttention)(query::AbstractArray, key::AbstractArray, value::AbstractArray; mask=nothing)

Multi-head attention implementation supporting arbitrary head dimensions independent of d_model.
"""
function (mha::MultiHeadAttention)(query::AbstractArray, key::AbstractArray, value::AbstractArray; mask=nothing)
    # Get dimensions
    d_model, seq_len_q, batch_size = size(query)
    _, seq_len_k, _ = size(key)
    
    # Get head dimensions
    h = mha.num_heads
    d_k = mha.head_dim
    total_dim = h * d_k
    
    # Project inputs to Q, K, V spaces - flatten once for efficiency
    q_flat = reshape(query, d_model, :)
    k_flat = reshape(key, d_model, :)
    v_flat = reshape(value, d_model, :)
    
    q_proj_flat = mha.W_q(q_flat)
    k_proj_flat = mha.W_k(k_flat)
    v_proj_flat = mha.W_v(v_flat)
    
    # Reshape projections to include head dimension
    q_proj = reshape(q_proj_flat, total_dim, seq_len_q, batch_size)
    k_proj = reshape(k_proj_flat, total_dim, seq_len_k, batch_size)
    v_proj = reshape(v_proj_flat, total_dim, seq_len_k, batch_size)
    
    # Process each batch using functional programming
    batch_outputs = map(1:batch_size) do b
        q_batch = view(q_proj, :, :, b)
        k_batch = view(k_proj, :, :, b)
        v_batch = view(v_proj, :, :, b)
        
        # Process heads - always use sequential processing for Zygote compatibility
        head_outputs = map(1:h) do head
            head_start = (head-1) * d_k + 1
            head_end = head * d_k
            
            q_head = view(q_batch, head_start:head_end, :)
            k_head = view(k_batch, head_start:head_end, :)
            v_head = view(v_batch, head_start:head_end, :)
            
            # Compute scaled dot-product attention
            scores = (q_head' * k_head) .* mha.scale
            
            # Apply mask if provided
            if mask !== nothing
                scores = scores .+ mask
            end
            
            # Apply softmax to get attention weights
            attention_weights = softmax(scores, dims=2)
            attention_weights = mha.dropout(attention_weights)
            
            # Apply attention to values
            v_head * attention_weights'
        end
        
        # Concatenate all heads for this batch (non-mutating)
        vcat(head_outputs...)
    end
    
    # Combine all batches (non-mutating)
    output_3d = cat(map(batch -> reshape(batch, total_dim, seq_len_q, 1), batch_outputs)..., dims=3)
    
    # Final projection from total_dim back to d_model
    output_flat = reshape(output_3d, total_dim, :)
    final_output_flat = mha.W_o(output_flat)
    
    return reshape(final_output_flat, d_model, seq_len_q, batch_size)
end

"""
    PositionalEncoding

Optimized transformer positional encoding layer with pre-computation.
"""
struct PositionalEncoding
    embedding::Matrix{Float32}
    
    function PositionalEncoding(max_len::Int, d_model::Int)
        # Pre-compute positional encoding matrix efficiently
        pe = zeros(Float32, d_model, max_len)
        position = reshape(1:max_len, 1, :)
        
        # Handle dimension calculation more carefully
        for i in 0:(d_model-1)
            # Calculate frequency based on position
            freq = exp(-(log(10000.0) * (i ÷ 2) / (d_model ÷ 2)))
            
            # Even indices get sine, odd indices get cosine
            if i % 2 == 0
                pe[i+1, :] = sin.(position .* freq)
            else
                pe[i+1, :] = cos.(position .* freq)
            end
        end
        
        new(pe)
    end
    
    # Constructor for functor reconstruction
    PositionalEncoding(embedding) = new(embedding)
end

# Add Flux functor declaration for PositionalEncoding
Flux.@functor PositionalEncoding

"""
    (pe::PositionalEncoding)(x)

Fully functional positional encoding application without mutations.
"""
function (pe::PositionalEncoding)(x)
    d_model, seq_len, batch_size = size(x)
    seq_len = min(seq_len, size(pe.embedding, 2))
    
    # Create result using functional approach (map + concatenate)
    result = cat(
        map(1:batch_size) do b
            # For each batch
            batch_result = cat(
                map(1:seq_len) do s
                    # For each position in sequence, add positional encoding
                    x[:, s, b] + pe.embedding[:, s]
                end...,
                dims=2
            )
            
            # If needed, pad with original values for positions beyond available encodings
            if seq_len < size(x, 2)
                cat(
                    batch_result,
                    x[:, (seq_len+1):end, b],
                    dims=2
                )
            else
                batch_result
            end
        end...,
        dims=3
    )
    
    return result
end

"""
    FeedForward

Standard feed-forward network used in transformer blocks.
"""
struct FeedForward
    W1::Dense
    W2::Dense
    dropout::Dropout
    
    function FeedForward(d_model::Int, d_ff::Int, dropout_rate::Float64=0.1)
        W1 = Dense(d_model, d_ff, relu)
        W2 = Dense(d_ff, d_model)
        dropout = Dropout(dropout_rate)
        
        new(W1, W2, dropout)
    end
    
    # Constructor for functor reconstruction
    FeedForward(W1, W2, dropout) = new(W1, W2, dropout)
end

# Add Flux functor declaration for FeedForward
Flux.@functor FeedForward

function (ff::FeedForward)(x::AbstractArray)
    # Get input dimensions
    d_model, seq_len, batch_size = size(x)
    
    # Reshape for dense layers
    x_flat = reshape(x, d_model, :)
    
    # Apply feed-forward network
    h = ff.W1(x_flat)
    h = ff.dropout(h)
    out = ff.W2(h)
    
    # Reshape back to original dimensions
    reshape(out, d_model, seq_len, batch_size)
end

"""
    TransformerEncoderLayer

Standard encoder layer that combines multi-head attention and feed-forward networks.
"""
struct TransformerEncoderLayer
    attention::MultiHeadAttention
    norm1::LayerNorm
    feed_forward::FeedForward
    norm2::LayerNorm
    dropout::Dropout
    
    # Constructor for functor reconstruction
    TransformerEncoderLayer(attention, norm1, feed_forward, norm2, dropout) = 
        new(attention, norm1, feed_forward, norm2, dropout)
end

# Add Flux functor declaration for TransformerEncoderLayer
Flux.@functor TransformerEncoderLayer

function TransformerEncoderLayer(d_model::Int, num_heads::Int, d_ff::Int, dropout_rate::Float64=0.1, head_dim::Int=0)
    attention = MultiHeadAttention(d_model, num_heads, head_dim, dropout_rate)
    norm1 = LayerNorm(d_model)
    feed_forward = FeedForward(d_model, d_ff, dropout_rate)
    norm2 = LayerNorm(d_model)
    dropout = Dropout(dropout_rate)
    TransformerEncoderLayer(attention, norm1, feed_forward, norm2, dropout)
end

# Forward pass with residual connections and normalization
function (layer::TransformerEncoderLayer)(x::AbstractArray, mask=nothing)
    # Multi-head attention with residual connection and layer norm
    attended = layer.attention(x; mask=mask)
    attended = layer.dropout(attended)
    x1 = layer.norm1(x .+ attended)
    
    # Feed-forward with residual connection and layer norm
    transformed = layer.feed_forward(x1)
    transformed = layer.dropout(transformed)
    layer.norm2(x1 .+ transformed)
end

"""
    ContinuousTransformerModel

Transformer model for continuous time series prediction using delay embeddings.
Processes delay embedding vectors directly with attention across embedding vectors.
"""
struct ContinuousTransformerModel
    input_projection::Dense
    transformer_layers::Vector{TransformerEncoderLayer}
    norm::LayerNorm
    output_projection::Dense
    
    function ContinuousTransformerModel(;
        input_dim::Int,
        output_dim::Int,
        d_model::Int,
        num_heads::Int,
        num_layers::Int,
        dropout_rate::Float32 = 0.1f0,
        d_ff::Union{Int,Nothing} = nothing
    )
        # Set feed-forward dimension if not provided (standard is 4x d_model)
        d_ff_actual = isnothing(d_ff) ? 4 * d_model : d_ff
        
        # Project input embedding to model dimension
        input_proj = Dense(input_dim, d_model)
        
        # Create transformer encoder layers without positional encoding
        layers = TransformerEncoderLayer[]
        for _ in 1:num_layers
            layer = TransformerEncoderLayer(d_model, num_heads, d_ff_actual, Float64(dropout_rate))
            push!(layers, layer)
        end
        
        # Final layer normalization
        norm = LayerNorm(d_model)
        
        # Project back to output dimension
        output_proj = Dense(d_model, output_dim)
        
        new(input_proj, layers, norm, output_proj)
    end
    
    # Constructor for reconstruction
    ContinuousTransformerModel(input_projection, transformer_layers, norm, output_projection) = 
        new(input_projection, transformer_layers, norm, output_projection)
end

Flux.@functor ContinuousTransformerModel

function (model::ContinuousTransformerModel)(x)
    # x shape: (embedding_dim, n_samples) or (embedding_dim, batch_size, n_batches)
    
    # Handle both 2D and 3D inputs
    if ndims(x) == 2
        # Original 2D case: (embedding_dim, n_samples)
        x_proj = model.input_projection(x)  # (d_model, n_samples)
        
        # Reshape for transformer: (d_model, n_samples, 1) - treat as single batch
        x_reshaped = reshape(x_proj, size(x_proj, 1), size(x_proj, 2), 1)
        
        # Pass through transformer encoder layers
        for layer in model.transformer_layers
            x_reshaped = layer(x_reshaped)
        end
        
        # Apply final layer normalization
        x_encoded = model.norm(x_reshaped)
        
        # Remove batch dimension: (d_model, n_samples)
        x_encoded = x_encoded[:, :, 1]
        
        # Project to output
        output = model.output_projection(x_encoded)  # (output_dim, n_samples)
        
    else
        # New 3D case: (embedding_dim, batch_size, n_batches)
        embedding_dim, batch_size, n_batches = size(x)
        
        # Reshape to 2D for projection: (embedding_dim, batch_size * n_batches)
        x_flat = reshape(x, embedding_dim, batch_size * n_batches)
        x_proj = model.input_projection(x_flat)  # (d_model, batch_size * n_batches)
        
        # Reshape back to 3D: (d_model, batch_size, n_batches)
        x_reshaped = reshape(x_proj, size(x_proj, 1), batch_size, n_batches)
        
        # Pass through transformer encoder layers
        for layer in model.transformer_layers
            x_reshaped = layer(x_reshaped)
        end
        
        # Apply final layer normalization
        x_encoded = model.norm(x_reshaped)
        
        # Reshape to 2D for output projection: (d_model, batch_size * n_batches)
        x_encoded_flat = reshape(x_encoded, size(x_encoded, 1), batch_size * n_batches)
        output_flat = model.output_projection(x_encoded_flat)  # (output_dim, batch_size * n_batches)
        
        # Reshape back to 3D: (output_dim, batch_size, n_batches)
        output = reshape(output_flat, size(output_flat, 1), batch_size, n_batches)
    end
    
    return output
end

"""
    denormalize_value(processor, normalized_value)

Convert a normalized value back to original scale using the processor's normalization parameters.
"""
function denormalize_value(processor::DelayEmbeddingProcessor, normalized_value::Float32)
    if processor.normalize
        # Assuming the processor has mean and std fields for normalization
        # This might need to be adjusted based on your actual DelayEmbeddingProcessor implementation
        return normalized_value * processor.std + processor.mean
    else
        return normalized_value
    end
end

"""
    get_embedding_data(processor, dim=1)

Get delay embedding data and targets for transformer training.
Returns (inputs, targets) where inputs is (embedding_dim, n_samples) and targets is (dim, n_samples).
For now, targets are kept in normalized scale (same as inputs).
"""
function get_embedding_data(processor::DelayEmbeddingProcessor, dim::Int=1)
    n_samples = size(processor.embedded_data, 1) - 1  # -1 because we predict next value
    
    # Inputs: delay embeddings (embedding_dim, n_samples)
    inputs = processor.embedded_data[1:n_samples, :]'
    
    # Targets: next values (keeping in normalized scale for now)
    # Take the first component of the next embedding (which represents the next value)
    targets = reshape(processor.embedded_data[2:n_samples+1, 1], 1, :)
    
    return Float32.(inputs), Float32.(targets)
end

"""
    train_continuous_transformer!(model, processor; kwargs...)

Train transformer model on delay embedding data.
"""
function train_continuous_transformer!(
    model::ContinuousTransformerModel,
    processor::DelayEmbeddingProcessor;
    epochs::Int = 100,
    batch_size::Int = 32,
    learning_rate::Float32 = 1f-3,
    early_stopping_patience::Int = 10,
    verbose::Bool = true,
    n_batches_per_epoch::Int = 50
)
    
    # Get training data
    inputs, targets = get_embedding_data(processor)
    
    # Split into train/validation
    n_total = size(inputs, 2)
    n_train = Int(round(0.8 * n_total))
    
    train_inputs = inputs[:, 1:n_train]
    train_targets = targets[:, 1:n_train]
    val_inputs = inputs[:, n_train+1:end]
    val_targets = targets[:, n_train+1:end]
    
    verbose && println("Training data: $(size(train_inputs, 2)) samples")
    verbose && println("Validation data: $(size(val_inputs, 2)) samples")
    
    # Setup optimizer
    optimizer = Flux.Adam(learning_rate)
    opt_state = Flux.setup(optimizer, model)
    
    # Training loop
    train_losses = Float32[]
    val_losses = Float32[]
    best_val_loss = Inf32
    patience_counter = 0
    
    for epoch in 1:epochs
        # Training - parallel batch processing
        n_possible_batches = div(size(train_inputs, 2) - batch_size + 1, 1)
        actual_n_batches = min(n_batches_per_epoch, n_possible_batches)
        
        # Randomly sample batch starting positions
        batch_starts = randperm(n_possible_batches)[1:actual_n_batches]
        batch_starts = [min(start, size(train_inputs, 2) - batch_size + 1) for start in batch_starts]
        
        # Create 3D tensors for parallel batch processing
        # Shape: (embedding_dim, batch_size, n_batches)
        batch_inputs = Array{Float32}(undef, size(train_inputs, 1), batch_size, actual_n_batches)
        batch_targets = Array{Float32}(undef, size(train_targets, 1), batch_size, actual_n_batches)
        
        # Fill the batch tensors
        for (i, batch_start) in enumerate(batch_starts)
            batch_end = min(batch_start + batch_size - 1, size(train_inputs, 2))
            batch_inputs[:, :, i] = train_inputs[:, batch_start:batch_end]
            batch_targets[:, :, i] = train_targets[:, batch_start:batch_end]
        end
        
        # Single forward and backward pass for all batches
        loss, grads = Flux.withgradient(model) do m
            predictions = m(batch_inputs)  # Process all batches in parallel
            Flux.mse(predictions, batch_targets)
        end
        
        Flux.update!(opt_state, model, grads[1])
        push!(train_losses, loss)
        
        # Validation
        val_loss = 0f0
        n_val_batches = 0
        
        for batch_start in 1:batch_size:size(val_inputs, 2)
            batch_end = min(batch_start + batch_size - 1, size(val_inputs, 2))
            
            val_batch_inputs = val_inputs[:, batch_start:batch_end]
            val_batch_targets = val_targets[:, batch_start:batch_end]
            
            predictions = model(val_batch_inputs)
            val_loss += Flux.mse(predictions, val_batch_targets)
            n_val_batches += 1
        end
        
        avg_val_loss = val_loss / n_val_batches
        push!(val_losses, avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss
            best_val_loss = avg_val_loss
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        if verbose && (epoch % 5 == 0 || epoch == 1)
            println("Epoch $epoch: Train Loss = $(round(loss, digits=6)), Val Loss = $(round(avg_val_loss, digits=6))")
        end
        
        if patience_counter >= early_stopping_patience
            verbose && println("Early stopping at epoch $epoch")
            break
        end
    end
    
    return model, train_losses, val_losses
end

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
data = generate_lorenz63_data(100000, tspan=(0.0, 5000.0))

# Extract the y-variable for time series prediction
y_data = data[:, 2]

println("Generated $(length(y_data)) data points")
println("Data range: [$(round(minimum(y_data), digits=3)), $(round(maximum(y_data), digits=3))]")

# Create delay embedding instead of discrete clusters
embedding_dim = 32  # Delay embedding dimension (m)
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
    println("  - Mean: $(round(norm_info.mean, digits=3))")
    println("  - Std: $(round(norm_info.std, digits=3))")
end

# ===================================================================
# 2. Model Configuration
# ===================================================================

println("\n3. Configuring transformer model...")

# Model hyperparameters - removed sequence_length as it's no longer needed
d_model = 64           # Model dimension
num_heads = 16          # Number of attention heads
num_layers = 1         # Number of transformer layers
dropout_rate = 0.1f0   # Dropout rate for regularization (Float32)

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

# Create the continuous transformer model
model = ContinuousTransformerModel(
    input_dim = embedding_dim,
    output_dim = 1,  # Predicting single next value
    d_model = d_model,
    num_heads = num_heads,
    num_layers = num_layers,
    dropout_rate = dropout_rate
)

# Count parameters
param_count = sum(length, Flux.params(model))
println("Total trainable parameters: $param_count")

# ===================================================================
# 3. Training and Evaluation
# ===================================================================

println("\n4. Training transformer model...")

# Train model with the new simplified API
model, train_losses, val_losses = train_continuous_transformer!(
    model, 
    processor;
    epochs = 500,
    batch_size = 32,
    learning_rate = 1f-4,
    early_stopping_patience = 50,
    verbose = true,
    n_batches_per_epoch = 50
)

println("Training completed successfully!")

##
plot(train_losses)