using Flux
using NNlib
using DifferentialEquations
using SciMLSensitivity

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
    NeuralODE

Neural ODE layer that takes latent_dim input and produces latent_dim output.
The input must already be projected to latent_dim dimensions.
Integrates the input for T time steps and returns a trajectory.
The neural network can have multiple internal layers specified by node_layers vector.
"""
struct NeuralODE
    neural_net::Chain     # Multi-layer dynamics in latent space
    T::Float32            # Integration time
    dt::Float32           # Time step size
    
    function NeuralODE(latent_dim::Int, node_layers::Vector{Int}=Int[], T::Float32=1.0f0, dt::Float32=0.1f0)
        # Build the neural network layers
        layers = []
        
        if isempty(node_layers)
            # Default: single layer with identity activation
            push!(layers, Dense(latent_dim, latent_dim, identity))
        else
            # First layer: latent_dim -> node_layers[1]
            push!(layers, Dense(latent_dim, node_layers[1], tanh))
            
            # Internal layers: node_layers[i] -> node_layers[i+1]
            for i in 1:(length(node_layers)-1)
                push!(layers, Dense(node_layers[i], node_layers[i+1], tanh))
            end
            
            # Final layer: node_layers[end] -> latent_dim (identity activation)
            push!(layers, Dense(node_layers[end], latent_dim, identity))
        end
        
        # Create the chain of layers
        neural_net = Chain(layers...)
        
        new(neural_net, T, dt)
    end
    
    # Constructor for functor reconstruction
    NeuralODE(neural_net, T, dt) = new(neural_net, T, dt)
end

# Add Flux functor declaration for NeuralODE
Flux.@functor NeuralODE

function (node::NeuralODE)(x_latent::AbstractArray)
    # x_latent should have shape (latent_dim, seq_len * batch_size)
    latent_dim, n_flat = size(x_latent)
    
    # Define the ODE function in latent space
    function ode_func(u, p, t)
        # u has shape (latent_dim, seq_len * batch_size)
        node.neural_net(u)
    end
    
    # Time span for integration - use a cleaner approach
    n_steps = max(1, Int(round(node.T / node.dt)))  # Ensure at least 1 step
    time_points = Float32.(0:n_steps) .* node.dt  # Create exactly n_steps+1 points
    tspan = (0.0f0, node.T)
    
    # Solve ODE for each time step we want to collect
    prob = ODEProblem(ode_func, x_latent, tspan)
    
    # Use QuadratureAdjoint which should be more widely available
    sol = solve(prob, Tsit5(), saveat=time_points, sensealg=QuadratureAdjoint())
    
    # Extract trajectory (excluding initial condition if we have enough points)
    if length(sol.u) > 1
        trajectory_latent = sol.u[2:end]  # Skip initial condition
        actual_n_steps = length(trajectory_latent)
    else
        trajectory_latent = sol.u  # Keep the single point if that's all we have
        actual_n_steps = 1
    end
    
    # Stack trajectory: (latent_dim, seq_len * batch_size, actual_n_steps)
    trajectory_latent_stacked = cat(trajectory_latent..., dims=3)
    
    # Final shape: (latent_dim, seq_len * batch_size, actual_n_steps)
    return trajectory_latent_stacked
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
function (layer::TransformerEncoderLayer)(x::AbstractArray, targets=nothing)
    # Multi-head attention with residual connection and layer norm
    attended = layer.attention(x)
    attended = layer.dropout(attended)
    x1 = layer.norm1(x .+ attended)
    
    # Feed-forward with residual connection and layer norm
    transformed = layer.feed_forward(x1)
    transformed = layer.dropout(transformed)
    x2 = layer.norm2(x1 .+ transformed)

    return x2  # Return full tensor, not flattened
end

"""
    NODELayer

Standard encoder layer that combines multi-head attention and feed-forward networks.
"""
struct NODELayer
        projection_in::Dense
        neural_ode::NeuralODE
        projection_out::Dense
    
    # Constructor for functor reconstruction
    NODELayer(projection_in, neural_ode, projection_out) = 
        new(projection_in, neural_ode, projection_out)
end

# Add Flux functor declaration for NODELayer
Flux.@functor NODELayer

function NODELayer(d_model::Int, latent_dim::Int, output_dim::Int, node_layers::Vector{Int}=Int[], T::Float32=1.0f0, dt::Float32=0.1f0)
    projection_in = Dense(d_model, latent_dim)
    neural_ode = NeuralODE(latent_dim, node_layers, T, dt)
    projection_out = Dense(latent_dim, output_dim)
    NODELayer(projection_in, neural_ode, projection_out)
end

# Forward pass with residual connections and normalization
function (layer::NODELayer)(x::AbstractArray)
    # Project to latent space
    x_latent = layer.projection_in(x)

    # Neural ODE integration to get trajectory
    trajectory_output = layer.neural_ode(x_latent)  # (latent_dim, seq_len * batch_size, n_steps)

    # Project back to output dimension
    trajectory_output = layer.projection_out(trajectory_output)  # (output_dim, seq_len * batch_size, n_steps)

    # Return the trajectory output
    return trajectory_output  # Return the trajectory output
end

"""
    ContinuousTransformerModel

Transformer model for continuous time series prediction using delay embeddings.
Processes delay embedding vectors directly with attention across embedding vectors.
"""
struct ContinuousTransformerModel
    input_projection::Dense
    transformer_layers::Vector{TransformerEncoderLayer}
    node_layer::NODELayer
    
    function ContinuousTransformerModel(;
        input_dim::Int,
        output_dim::Int,
        d_model::Int,
        latent_dim::Int,
        num_heads::Int,
        num_layers::Int,
        dropout_rate::Float32 = 0.1f0,
        d_ff::Union{Int,Nothing} = nothing,
        node_layers::Vector{Int} = Int[],
        T::Float32 = 1.0f0,
        dt::Float32 = 0.1f0
    )
        # Set feed-forward dimension if not provided (standard is 4x d_model)
        d_ff_actual = isnothing(d_ff) ? 4 * d_model : d_ff
        
        # Project input embedding to model dimension
        input_proj = Dense(input_dim, d_model)
        
        # Create transformer encoder layers without positional encoding
        layers = TransformerEncoderLayer[]
        for _ in 1:num_layers
            # Fix: Use correct constructor parameters
            layer = TransformerEncoderLayer(d_model, num_heads, d_ff_actual, Float64(dropout_rate))
            push!(layers, layer)
        end
        
        # Create NODELayer for final projection to output dimension
        node_layer = NODELayer(d_model, latent_dim, output_dim, node_layers, T, dt)

        # Create the model with input projection, transformer layers, and NODE layer
        new(input_proj, layers, node_layer)
    end
    
    # Constructor for reconstruction
    ContinuousTransformerModel(input_projection, transformer_layers, node_layer) = 
        new(input_projection, transformer_layers, node_layer)
end

Flux.@functor ContinuousTransformerModel

function (model::ContinuousTransformerModel)(x)
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

    # Fix: Flatten for NODE layer input
    x_proj_flat = reshape(x_proj, size(x_proj, 1), :)  # (d_model, seq_len * n_batches)
    
    # Apply final NODE layer
    x_encoded = model.node_layer(x_proj_flat) # (output_dim, seq_len * n_batches, n_steps)

    # Fix: Correct reshaping for final output
    output_dim, _, n_steps = size(x_encoded)
    x_encoded_reshaped = reshape(x_encoded, output_dim, seq_len, n_batches, n_steps)

    # Return the final output
    return x_encoded_reshaped  # (output_dim, seq_len, n_batches, n_steps)
end