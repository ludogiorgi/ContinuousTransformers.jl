import Flux
import Flux: Dense, Dropout, LayerNorm, softmax, relu
using LinearAlgebra
using Random
using Statistics
using NNlib
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
        # Create delay embedding: [y_{i+embedding_dim-1}, y_{i+embedding_dim-2}, ..., y_i]
        # This means inputs[1, i] = y_{i+embedding_dim-1} (newest)
        # and inputs[end, i] = y_i (oldest)
        for j in 1:embedding_dim
            inputs[j, i] = normalized_data[i + embedding_dim - j]
        end
        # Target: y_{i+embedding_dim} (the next value after the embedding window)
        targets[1, i] = normalized_data[i + embedding_dim]
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

Convert a normalized value back to original scale using the processor's normalization parameters.
"""
function denormalize_value(processor::DelayEmbeddingProcessor, normalized_value::Float32)
    if processor.normalize
        # Use the correct field names from DelayEmbeddingProcessor struct
        return normalized_value * processor.data_std + processor.data_mean
    else
        return normalized_value
    end
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
Modified for parallel batch and head processing.
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
        actual_head_dim = head_dim <= 0 ? div(d_model, num_heads) : head_dim
        
        if head_dim <= 0 && d_model % num_heads != 0
            @warn "d_model ($d_model) is not divisible by num_heads ($num_heads). Using head_dim = $(actual_head_dim)"
        end
        
        scale = Float32(1 / sqrt(actual_head_dim))
        
        total_dim = num_heads * actual_head_dim
        W_q = Dense(d_model, total_dim)
        W_k = Dense(d_model, total_dim)
        W_v = Dense(d_model, total_dim)
        W_o = Dense(total_dim, d_model)
        dropout = Dropout(dropout_rate)
        
        new(num_heads, actual_head_dim, W_q, W_k, W_v, W_o, dropout, scale)
    end

    MultiHeadAttention(num_heads, head_dim, W_q, W_k, W_v, W_o, dropout, scale) = 
        new(num_heads, head_dim, W_q, W_k, W_v, W_o, dropout, scale)
end

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

Multi-head attention implementation with parallel batch and head processing.
Input tensors are expected in shape (features, sequence_length, batch_size).
"""
function (mha::MultiHeadAttention)(query::AbstractArray, key::AbstractArray, value::AbstractArray; mask=nothing)
    # Get input dimensions: (d_model, seq_len, batch_size)
    d_model, seq_len_q, batch_size = size(query)
    _, seq_len_k, _ = size(key)
    _, seq_len_v, _ = size(value) # seq_len_v should be same as seq_len_k

    # Useful dimensions
    H = mha.num_heads    # Number of heads
    D_h = mha.head_dim   # Dimension of each head
    total_dim = H * D_h  # Total dimension after projection (num_heads * head_dim)

    # 1. Project inputs to Q, K, V spaces
    # Reshape for Dense layer: (d_model, seq_len * batch_size)
    q_flat = reshape(query, d_model, seq_len_q * batch_size)
    k_flat = reshape(key, d_model, seq_len_k * batch_size)
    v_flat = reshape(value, d_model, seq_len_v * batch_size)

    q_proj_flat = mha.W_q(q_flat) # (total_dim, seq_len_q * batch_size)
    k_proj_flat = mha.W_k(k_flat) # (total_dim, seq_len_k * batch_size)
    v_proj_flat = mha.W_v(v_flat) # (total_dim, seq_len_v * batch_size)

    # Reshape back to (total_dim, seq_len, batch_size)
    q_proj = reshape(q_proj_flat, total_dim, seq_len_q, batch_size)
    k_proj = reshape(k_proj_flat, total_dim, seq_len_k, batch_size)
    v_proj = reshape(v_proj_flat, total_dim, seq_len_v, batch_size)

    # 2. Reshape Q, K, V for batched head processing
    # Current shape: (H*D_h, seq_len, B)
    # Target shape for batched_mul: (D_h, seq_len, H*B)
    
    # Q: (D_h, H, L_q, B) -> permute to (D_h, L_q, H, B) -> reshape to (D_h, L_q, H*B)
    q_reshaped = reshape(q_proj, D_h, H, seq_len_q, batch_size)
    q_permuted = permutedims(q_reshaped, (1, 3, 2, 4))
    Q_batched = reshape(q_permuted, D_h, seq_len_q, H * batch_size)

    # K: (D_h, H, L_k, B) -> permute to (D_h, L_k, H, B) -> reshape to (D_h, L_k, H*B)
    k_reshaped = reshape(k_proj, D_h, H, seq_len_k, batch_size)
    k_permuted = permutedims(k_reshaped, (1, 3, 2, 4))
    K_batched = reshape(k_permuted, D_h, seq_len_k, H * batch_size)

    # V: (D_h, H, L_v, B) -> permute to (D_h, L_v, H, B) -> reshape to (D_h, L_v, H*B)
    v_reshaped = reshape(v_proj, D_h, H, seq_len_v, batch_size)
    v_permuted = permutedims(v_reshaped, (1, 3, 2, 4))
    V_batched = reshape(v_permuted, D_h, seq_len_v, H * batch_size)
    
    # 3. Scaled Dot-Product Attention
    # Q_batched: (D_h, L_q, H*B)
    # K_batched: (D_h, L_k, H*B)
    # We want scores: (L_q, L_k, H*B)
    # Achieved by Q_T * K where Q_T is (L_q, D_h, H*B)
    Q_T_batched = permutedims(Q_batched, (2, 1, 3)) # (L_q, D_h, H*B)
    
    # scores = batched_mul(A, B) where A=(m,k,N), B=(k,n,N) -> (m,n,N)
    scores = NNlib.batched_mul(Q_T_batched, K_batched) # (L_q, L_k, H*B)
    scores = scores .* mha.scale

    # 4. Apply mask (if provided)
    if mask !== nothing
        # mask is (L_q, L_k). It will broadcast to (L_q, L_k, H*B).
        scores = scores .+ mask
    end

    # 5. Apply softmax to get attention weights
    # Softmax over L_k (keys) for each query position
    attention_weights = Flux.softmax(scores, dims=2) # (L_q, L_k, H*B)
    attention_weights = mha.dropout(attention_weights)

    # 6. Apply attention to Values
    # V_batched: (D_h, L_v, H*B) where L_v = L_k
    # attention_weights: (L_q, L_k, H*B)
    # We need V * AW_T where AW_T is (L_k, L_q, H*B)
    AW_T_batched = permutedims(attention_weights, (2, 1, 3)) # (L_k, L_q, H*B)
    
    # attended_values = batched_mul(A,B) where A=(m,k,N), B=(k,n,N) -> (m,n,N)
    attended_values = NNlib.batched_mul(V_batched, AW_T_batched) # (D_h, L_q, H*B)

    # 7. Concatenate heads (reshape back)
    # attended_values is (D_h, L_q, H*B)
    # Target shape: (total_dim, L_q, B) which is (H*D_h, L_q, B)
    
    # (D_h, L_q, H, B)
    attended_reshaped = reshape(attended_values, D_h, seq_len_q, H, batch_size)
    # (D_h, H, L_q, B)
    attended_permuted = permutedims(attended_reshaped, (1, 3, 2, 4))
    # (H*D_h, L_q, B) which is (total_dim, L_q, B)
    context_layer = reshape(attended_permuted, total_dim, seq_len_q, batch_size)

    # 8. Final linear projection
    # Reshape for Dense layer: (total_dim, seq_len_q * batch_size)
    context_layer_flat = reshape(context_layer, total_dim, seq_len_q * batch_size)
    final_output_flat = mha.W_o(context_layer_flat) # (d_model, seq_len_q * batch_size)
    
    # Reshape back to (d_model, seq_len_q, batch_size)
    final_output = reshape(final_output_flat, d_model, seq_len_q, batch_size)
    
    return final_output
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

Efficient and Zygote-friendly positional encoding application.
"""
function (pe::PositionalEncoding)(x::AbstractArray{T, 3}) where T # x is (d_model, seq_len, batch_size)
    d_model, seq_len_in, batch_size = size(x)
    max_encode_len = size(pe.embedding, 2) # pe.embedding is (d_model, max_len)
    
    # Determine the length of positional encoding to apply
    len_to_apply = min(seq_len_in, max_encode_len)
    
    # Create the additive positional encoding term.
    # This will be added to x.
    # We construct it to be the same size as x for direct addition.
    additive_pe = zeros(T, d_model, seq_len_in, batch_size) # Initialize with zeros
    
    if len_to_apply > 0
        # Get the relevant slice of positional embeddings
        pe_slice_to_add = view(pe.embedding, :, 1:len_to_apply) # Shape: (d_model, len_to_apply)
        
        # Broadcast this slice to all batch items in additive_pe for the applicable sequence length
        # view(additive_pe, :, 1:len_to_apply, :) gets shape (d_model, len_to_apply, batch_size)
        # pe_slice_to_add needs to be reshaped or broadcast appropriately.
        # Easiest is to assign it to each batch item's slice.
        for b_idx in 1:batch_size
            view(additive_pe, :, 1:len_to_apply, b_idx) .= pe_slice_to_add
        end
        # If seq_len_in > len_to_apply, the rest of additive_pe remains zero, so x is unchanged there.
    end
    
    return x .+ additive_pe # Final out-of-place addition
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
    # Handle both 2D and 3D inputs
    if ndims(x) == 2
        # x shape: (embedding_dim, seq_len) - single batch case
        # Project inputs to model dimension
        x_proj = model.input_projection(x)  # (d_model, seq_len)
        
        # Reshape for transformer: (d_model, seq_len, 1) - treat as single batch
        x_reshaped = reshape(x_proj, size(x_proj, 1), size(x_proj, 2), 1)
        
        # Pass through transformer encoder layers
        for layer in model.transformer_layers
            x_reshaped = layer(x_reshaped)
        end
        
        # Apply final layer normalization
        x_encoded = model.norm(x_reshaped)
        
        # Remove batch dimension: (d_model, seq_len)
        x_encoded = x_encoded[:, :, 1]
        
        # Project to output for each sample
        output = model.output_projection(x_encoded)  # (output_dim, seq_len)
        
        return output
    else
        # x shape: (embedding_dim, seq_len, n_batches) - multiple batch case
        embedding_dim, seq_len, n_batches = size(x)
        
        # Reshape to process all batches at once: (embedding_dim, seq_len * n_batches)
        x_flat = reshape(x, embedding_dim, seq_len * n_batches)
        
        # Project inputs to model dimension
        x_proj_flat = model.input_projection(x_flat)  # (d_model, seq_len * n_batches)
        
        # Reshape back to include batch dimension: (d_model, seq_len, n_batches)
        x_proj = reshape(x_proj_flat, size(x_proj_flat, 1), seq_len, n_batches)
        
        # Pass through transformer encoder layers
        for layer in model.transformer_layers
            x_proj = layer(x_proj)
        end
        
        # Apply final layer normalization
        x_encoded = model.norm(x_proj)
        
        # Flatten for output projection: (d_model, seq_len * n_batches)
        x_encoded_flat = reshape(x_encoded, size(x_encoded, 1), seq_len * n_batches)
        
        # Project to output
        output_flat = model.output_projection(x_encoded_flat)  # (output_dim, seq_len * n_batches)
        
        # Reshape back to batch format: (output_dim, seq_len, n_batches)
        output = reshape(output_flat, size(output_flat, 1), seq_len, n_batches)

        return output
    end
end

"""
    denormalize_value(processor, normalized_value)

Convert a normalized value back to original scale using the processor's normalization parameters.
"""
function denormalize_value(processor::DelayEmbeddingProcessor, normalized_value::Float32)
    if processor.normalize
        # Use the correct field names from DelayEmbeddingProcessor struct
        return normalized_value * processor.data_std + processor.data_mean
    else
        return normalized_value
    end
end

"""
    get_embedding_data(processor, dim=1)

Get delay embedding data and targets for transformer training.
Returns (inputs, targets) where inputs is (embedding_dim, n_samples) and targets is (dim, n_samples).
The delay embedding is structured so that inputs[end, N] = targets[1, N-1].
"""
function get_embedding_data(processor::DelayEmbeddingProcessor, dim::Int=1)
    normalized_data = processor.normalized_data
    embedding_dim = processor.embedding_dim
    
    # We need at least embedding_dim + 1 points to create one input-target pair
    n_samples = length(normalized_data) - embedding_dim
    if n_samples <= 0
        error("Not enough data for embedding dimension $embedding_dim")
    end
    
    # Input matrix: (embedding_dim, n_samples)
    inputs = Array{Float32}(undef, embedding_dim, n_samples)
    # Target matrix: (1, n_samples)
    targets = Array{Float32}(undef, 1, n_samples)
    
    for i in 1:n_samples
        # Create delay embedding: [y_{i+embedding_dim-1}, y_{i+embedding_dim-2}, ..., y_i]
        # This means inputs[1, i] = y_{i+embedding_dim-1} (newest)
        # and inputs[end, i] = y_i (oldest)
        for j in 1:embedding_dim
            inputs[j, i] = normalized_data[i + j - 1]
        end
        # Target: y_{i+embedding_dim} (the next value after the embedding window)
        targets[1, i] = normalized_data[i + embedding_dim]
    end
    
    return inputs, targets
end

"""
    train_continuous_transformer!(model, processor; kwargs...)

Train transformer model on delay embedding data with optimized batching for training AND validation.
"""
function train_continuous_transformer!(
    model::ContinuousTransformerModel,
    processor::DelayEmbeddingProcessor;
    epochs::Int = 100,
    seq_len::Int = 32, # Sequence length for training batches
    val_seq_len::Int = 256, # Sequence length for validation batches (can be larger than train_seq_len)
    learning_rate::Float32 = 1f-3,
    early_stopping_patience::Int = 10,
    verbose::Bool = true,
    n_training_steps_per_epoch::Int = 100, 
    training_batch_size::Int = 4 
)
    
    inputs, targets = get_embedding_data(processor) 
    
    n_total_samples = size(inputs, 2)
    n_train_samples = Int(round(0.8 * n_total_samples))
    
    # Use views to avoid copying large datasets initially
    train_inputs_full = view(inputs, :, 1:n_train_samples)
    train_targets_full = view(targets, :, 1:n_train_samples)
    val_inputs_full = view(inputs, :, n_train_samples+1:n_total_samples)
    val_targets_full = view(targets, :, n_train_samples+1:n_total_samples)
    
    input_dim = size(train_inputs_full, 1)
    output_dim = size(train_targets_full, 1)

    verbose && println("Training data: $(size(train_inputs_full, 2)) samples, $(input_dim) features")
    verbose && println("Validation data: $(size(val_inputs_full, 2)) samples, $(input_dim) features")
    
    optimizer = Flux.Adam(learning_rate)
    opt_state = Flux.setup(optimizer, model)
    
    train_losses = Float32[]
    val_losses = Float32[]
    best_val_loss = Inf32
    patience_counter = 0

    batch_inputs_alloc = Array{Float32}(undef, input_dim, seq_len, training_batch_size)
    batch_targets_alloc = Array{Float32}(undef, output_dim, seq_len, training_batch_size)
    
    for epoch in 1:epochs
        # Training
        epoch_train_loss = 0f0
        actual_training_steps_this_epoch = 0
        
        if size(train_inputs_full, 2) < seq_len
            @warn "Not enough training data for sequence length $seq_len. Skipping epoch $epoch."
            # Optionally push a marker loss or handle as error
            push!(train_losses, NaN32) 
            push!(val_losses, NaN32)   
            continue
        end

        for step in 1:n_training_steps_per_epoch
            for b_idx in 1:training_batch_size 
                start_idx = rand(1:(size(train_inputs_full, 2) - seq_len + 1))
                end_idx = start_idx + seq_len - 1
                
                copyto!(view(batch_inputs_alloc, :, :, b_idx), view(train_inputs_full, :, start_idx:end_idx))
                copyto!(view(batch_targets_alloc, :, :, b_idx), view(train_targets_full, :, start_idx:end_idx))
            end
            
            loss, grads = Flux.withgradient(model) do m
                predictions = m(batch_inputs_alloc) 
                Flux.mse(predictions, batch_targets_alloc)
            end
            
            if isnan(loss) || isinf(loss)
                @warn "Training loss is $loss at epoch $epoch, step $step. Skipping update."
                # Potentially log parameters or inputs if this happens frequently
                continue
            end

            Flux.update!(opt_state, model, grads[1])
            epoch_train_loss += loss
            actual_training_steps_this_epoch += 1
        end 
        
        avg_train_loss = actual_training_steps_this_epoch > 0 ? (epoch_train_loss / actual_training_steps_this_epoch) : 0f0
        push!(train_losses, avg_train_loss)
        
        # Validation (with batching)
        current_val_loss_total = 0f0
        num_val_batches = 0
        
        if size(val_inputs_full, 2) > 0
            if size(val_inputs_full, 2) < val_seq_len
                 @warn "Not enough validation data for val_seq_len $val_seq_len. Processing what's available."
            end

            for val_batch_start_idx in 1:val_seq_len:size(val_inputs_full, 2)
                val_batch_end_idx = min(val_batch_start_idx + val_seq_len - 1, size(val_inputs_full, 2))
                
                # Ensure the chunk is not empty (can happen if val_inputs_full size < val_seq_len initially)
                if val_batch_start_idx > val_batch_end_idx
                    continue
                end

                current_val_input_chunk = view(val_inputs_full, :, val_batch_start_idx:val_batch_end_idx)
                current_val_target_chunk = view(val_targets_full, :, val_batch_start_idx:val_batch_end_idx)
                
                # Model's 2D input path expects (features, sequence_length)
                # It will internally reshape to (features, sequence_length, 1)
                val_predictions_chunk = model(current_val_input_chunk) 
                
                loss_chunk = Flux.mse(val_predictions_chunk, current_val_target_chunk)
                if isnan(loss_chunk) || isinf(loss_chunk)
                    @warn "Validation loss chunk is $loss_chunk at epoch $epoch. Skipping this chunk."
                    continue
                end
                current_val_loss_total += loss_chunk
                num_val_batches += 1
            end
            
            current_epoch_val_loss = num_val_batches > 0 ? (current_val_loss_total / num_val_batches) : Inf32
        else
            current_epoch_val_loss = Inf32 
        end
        push!(val_losses, current_epoch_val_loss)
        
        # Early stopping
        if current_epoch_val_loss < best_val_loss
            best_val_loss = current_epoch_val_loss
            patience_counter = 0
            # Optionally save best model here
            # Example: Flux.Optimise.save("./best_model.bson", model)
        else
            patience_counter += 1
        end
        
        if verbose && (epoch % 5 == 0 || epoch == 1 || epoch == epochs)
            println("Epoch $epoch/$epochs: Train Loss = $(round(avg_train_loss, digits=6)), Val Loss = $(round(current_epoch_val_loss, digits=6)) (Best: $(round(best_val_loss, digits=6)))")
        end
        
        if patience_counter >= early_stopping_patience
            verbose && println("Early stopping at epoch $epoch due to patience.")
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
    println("  - Mean: $(round(norm_info.mean, digits=3))")
    println("  - Std: $(round(norm_info.std, digits=3))")
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
##


inputs, targets = get_embedding_data(processor) 

inputs[end-1, 10]
targets[1,8]


plot!(targets[1, 1:9], label="Target", color=:red)

##
# ===================================================================
# 3. Training and Evaluation
# ===================================================================

println("\n4. Training transformer model...")

model, train_losses, val_losses = train_continuous_transformer!(
    model, 
    processor;
    epochs = 60,
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

inputs, targets = get_embedding_data(processor) 
    
n_total_samples = size(inputs, 2)
n_train_samples = Int(round(0.8 * n_total_samples))
    
# Use views to avoid copying large datasets initially
train_inputs_full = view(inputs, :, 1:n_train_samples)
train_targets_full = view(targets, :, 1:n_train_samples)
val_inputs_full = view(inputs, :, n_train_samples+1:n_total_samples)
val_targets_full = view(targets, :, n_train_samples+1:n_total_samples)

##

"""
    predict_next_values(model, initial_input, n_preds)

Recursively predict the next n_preds values using the transformer model.
Updated to handle the corrected delay embedding structure.
"""
function predict_next_values(model::ContinuousTransformerModel, initial_input::AbstractArray, n_preds::Int)
    input_dim, seq_len, batch_size = size(initial_input)
    
    if batch_size != 1
        error("This function expects batch_size = 1, got $batch_size")
    end
    
    # Store predictions
    predictions = Float32[]
    
    # Current input buffer - copy to avoid mutating original
    current_input = copy(initial_input)
    
    for step in 1:n_preds
        # Get prediction for the last time step
        prediction = model(current_input)[end, end, 1]  # Extract scalar prediction
        
        # Store normalized prediction
        push!(predictions, prediction)
        
        # Update input for next prediction
        # Shift the sequence left and add the new prediction
        new_input = similar(current_input)
        
        # Shift existing sequences
        new_input[:, 1:end-1, 1] = current_input[:, 2:end, 1]
        
        # For the last time step, create new delay embedding
        # Shift the delay embedding: newest becomes second newest, etc.
        new_input[1:end-1, end, 1] = current_input[2:end, end, 1]
        # Add prediction as the newest value in the delay embedding
        new_input[end, end, 1] = prediction
        
        current_input = new_input
    end
    
    return predictions
end

"""
    generate_ensemble_predictions(model, val_inputs_full, val_targets_full; n_ens=100, seq_len=100, n_preds=50, seed=nothing)

Generate ensemble predictions from multiple random starting points in the validation set.

# Arguments
- `model`: Trained ContinuousTransformerModel
- `val_inputs_full`: Validation input data 
- `val_targets_full`: Validation target data
- `n_ens`: Number of ensemble trajectories to generate (default: 100)
- `seq_len`: Sequence length for input windows (default: 100)  
- `n_preds`: Number of future steps to predict (default: 50)
- `seed`: Random seed for reproducible test selection (default: nothing for random)

# Returns
- `predictions`: Array of size (n_preds, n_ens) containing predicted trajectories
- `observations`: Array of size (n_preds, n_ens) containing ground truth trajectories
- `input_end_indices`: Vector of length n_ens containing the input end indices used
"""
function generate_ensemble_predictions(model::ContinuousTransformerModel, val_inputs_full, val_targets_full; 
                                     n_ens::Int=100, seq_len::Int=100, n_preds::Int=50, seed=nothing)
    
    # Set random seed if specified
    if seed !== nothing
        Random.seed!(seed)
    end
    
    n_val_samples = size(val_inputs_full, 2)
    
    # Generate COMPLETELY FIXED input END indices that don't depend on ANY parameters
    conservative_min_end = 500  # Leave room for large seq_len
    conservative_max_end = n_val_samples - n_preds - 100  # Leave room for predictions
    
    if conservative_max_end < conservative_min_end
        conservative_max_end = n_val_samples ÷ 2
        conservative_min_end = conservative_max_end ÷ 2
    end
    
    fixed_input_end_indices = [rand(conservative_min_end:conservative_max_end) for _ in 1:n_ens]
    
    # Initialize output arrays
    predictions = Array{Float32}(undef, n_preds, n_ens)
    observations = Array{Float32}(undef, n_preds, n_ens)
    successful_indices = Int[]
    
    for ens_idx in 1:n_ens
        # Use the completely FIXED input end index
        fixed_input_end_idx = fixed_input_end_indices[ens_idx]
        
        # Calculate input window based on seq_len - same end, different start
        input_end_idx = fixed_input_end_idx
        input_start_idx = input_end_idx - seq_len + 1
        
        # Check if we have enough data
        if input_start_idx < 1 || input_end_idx + n_preds > n_val_samples
            @warn "Skipping ensemble $ens_idx: not enough data for fixed end $fixed_input_end_idx with seq_len=$seq_len and n_preds=$n_preds"
            continue
        end
        
        # Create test input
        test_input = reshape(val_inputs_full[:, input_start_idx:input_end_idx], 
                           size(val_inputs_full, 1), seq_len, 1)
        
        # Make predictions
        pred_trajectory = predict_next_values(model, test_input, n_preds)
        
        # Get ground truth - the targets corresponding to predictions after input_end_idx
        target_start_idx = input_end_idx
        target_end_idx = target_start_idx + n_preds - 1
        obs_trajectory = val_targets_full[1, target_start_idx:target_end_idx]
        
        # Store results
        predictions[:, ens_idx] = pred_trajectory
        observations[:, ens_idx] = obs_trajectory
        push!(successful_indices, ens_idx)
    end
    
    # Return only successful predictions
    n_successful = length(successful_indices)
    if n_successful < n_ens
        @warn "Only $n_successful out of $n_ens ensemble members were successful"
        return predictions[:, successful_indices], observations[:, successful_indices], fixed_input_end_indices[successful_indices]
    end
    
    return predictions, observations, fixed_input_end_indices
end

"""
    combined_prediction_analysis(model, val_inputs_full, val_targets_full; 
                                n_ens=50, seq_len=32, n_preds_example=100, 
                                max_n_preds=150, n_pred_steps=15, seed=42)

Combined analysis function that generates ensemble trajectory plots and RMSE scaling analysis.

# Arguments
- `model`: Trained ContinuousTransformerModel
- `val_inputs_full`: Validation input data 
- `val_targets_full`: Validation target data
- `n_ens`: Number of ensemble trajectories to generate (default: 50)
- `seq_len`: Sequence length for input windows (default: 32)
- `n_preds_example`: Number of prediction steps for example trajectories (default: 100)
- `max_n_preds`: Maximum prediction horizon for RMSE analysis (default: 150)
- `n_pred_steps`: Number of different prediction horizons to test for RMSE (default: 15)
- `seed`: Random seed for reproducible results (default: 42)

# Returns
- `combined_plot`: Combined plot with trajectory examples (top 2 rows) and RMSE scaling (bottom row)
- `ensemble_predictions`: Prediction trajectories for examples
- `ensemble_observations`: Observation trajectories for examples  
- `n_pred_horizons`: Prediction horizon values tested for RMSE
- `rmse_scaling`: RMSE values for each horizon
"""
function combined_prediction_analysis(model::ContinuousTransformerModel, val_inputs_full, val_targets_full; 
                                    n_ens::Int=50, seq_len::Int=32, n_preds_example::Int=100, 
                                    max_n_preds::Int=150, n_pred_steps::Int=15, seed=42)
    
    println("\n=== Combined Prediction Analysis ===")
    
    # Generate ensemble predictions for example trajectories
    println("Generating ensemble predictions for example trajectories...")
    ensemble_predictions, ensemble_observations, ensemble_indices = generate_ensemble_predictions(
        model, val_inputs_full, val_targets_full; 
        n_ens=n_ens, seq_len=seq_len, n_preds=n_preds_example, seed=seed
    )
    
    # Analyze RMSE scaling with prediction horizon
    println("Analyzing RMSE scaling with prediction horizon...")
    n_pred_horizons, rmse_scaling, rmse_scaling_std, _ = analyze_prediction_horizon_scaling(
        model, val_inputs_full, val_targets_full;
        n_ens=n_ens, seq_len=seq_len, max_n_preds=max_n_preds, n_pred_steps=n_pred_steps, seed=seed
    )
    
    # Create individual trajectory plots (first 6 trajectories)
    n_plots_to_use = min(6, size(ensemble_predictions, 2))
    pred_time = 1:n_preds_example
    trajectory_plots = []
    
    for plot_idx in 1:n_plots_to_use
        # Show legend only in the first panel
        legend_setting = plot_idx == 1 ? :topright : false
        
        # Create individual trajectory plot
        p = plot(title="Trajectory $plot_idx (End: $(ensemble_indices[plot_idx]))", 
                xlabel="Prediction Step", ylabel="Value", legend=legend_setting,
                titlefontsize=10)
        
        # Plot predictions
        plot!(p, pred_time, ensemble_predictions[:, plot_idx], 
              label="Predictions", color=:red, linewidth=2, marker=:circle, markersize=3)
        
        # Plot ground truth
        plot!(p, pred_time, ensemble_observations[:, plot_idx], 
              label="Ground Truth", color=:green, linewidth=2, marker=:square, markersize=3)
        
        push!(trajectory_plots, p)
    end
    
    # Create RMSE scaling plot (spans full width)
    rmse_plot = plot(n_pred_horizons, rmse_scaling, 
                    ribbon=rmse_scaling_std,
                    xlabel="Prediction Horizon (steps)", 
                    ylabel="RMSE",
                    title="Prediction Error Scaling with Horizon",
                    label="RMSE ± σ",
                    linewidth=3,
                    marker=:circle,
                    markersize=5,
                    legend=:topleft,
                    grid=true,
                    titlefontsize=12)
    
    # Create combined layout: 2 rows of 3 trajectory plots + 1 row spanning all columns for RMSE
    # Use a custom layout with the RMSE plot spanning the full width
    l = @layout [a{0.33w} b{0.33w} c{0.33w}
                 d{0.33w} e{0.33w} f{0.33w}
                 g{1.0w}]
    
    combined_plot = plot(trajectory_plots[1], trajectory_plots[2], trajectory_plots[3],
                        trajectory_plots[4], trajectory_plots[5], trajectory_plots[6],
                        rmse_plot,
                        layout=l, 
                        size=(1200, 900),
                        plot_title="Transformer Prediction Analysis")
    
    # Print summary statistics
    println("\n=== Analysis Summary ===")
    println("Example trajectories:")
    println("  - Generated $(size(ensemble_predictions, 2)) ensemble trajectories")
    println("  - Each trajectory has $(size(ensemble_predictions, 1)) prediction steps")
    
    println("\nRMSE scaling analysis:")
    println("  - Shortest horizon ($(n_pred_horizons[1]) steps): RMSE = $(round(rmse_scaling[1], digits=4))")
    println("  - Longest horizon ($(n_pred_horizons[end]) steps): RMSE = $(round(rmse_scaling[end], digits=4))")
    println("  - RMSE growth factor: $(round(rmse_scaling[end] / rmse_scaling[1], digits=2))x")
    
    # Calculate average growth rate
    log_growth_rate = (log(rmse_scaling[end]) - log(rmse_scaling[1])) / (n_pred_horizons[end] - n_pred_horizons[1])
    println("  - Exponential growth rate: $(round(log_growth_rate, digits=6)) per step")
    
    return combined_plot, ensemble_predictions, ensemble_observations, n_pred_horizons, rmse_scaling
end

# Run combined prediction analysis
println("\n6. Running combined prediction analysis...")
combined_plot, ensemble_preds, ensemble_obs, horizons, rmse_values = combined_prediction_analysis(
    model, val_inputs_full, val_targets_full;
    n_ens=50, seq_len=1, n_preds_example=100, max_n_preds=150, n_pred_steps=15, seed=42
)

display(combined_plot)


