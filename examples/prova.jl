using Flux
using Random
using Plots
using Statistics
import Flux: Dense, Dropout, LayerNorm, softmax, params, update!, mse
using LinearAlgebra # For potential use of transpose, though direct views/reshapes are often preferred
using DifferentialEquations # Not strictly needed for the model definition part if data is pre-generated

# Add threading capabilities (though not explicitly used in the model forward pass here)
# using Base.Threads

# --- Lorenz Data Generation (Unchanged from your original) ---
function lorenz63!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

function generate_lorenz63_data(n_points::Int; tspan=(0.0, 100.0), dt=nothing, σ=10.0, ρ=28.0, β=8/3)

    u0 = [1.0, 1.0, 1.0]
    p_lorenz = [σ, ρ, β] 
    
    actual_dt = dt === nothing ? (tspan[2] - tspan[1]) / (n_points - 1) : dt
    t_eval = range(tspan[1], tspan[2], length=n_points)
    
    prob = ODEProblem(lorenz63!, u0, tspan, p_lorenz)
    sol = solve(prob, Tsit5(), saveat=t_eval, reltol=1e-8, abstol=1e-8)
    
    data = hcat(sol.u...)'
    return Float32.(data)
end

function generate_lorenz63_data(n_points::Int, tspan::Tuple)
    return generate_lorenz63_data(n_points; tspan=tspan)
end

function normalize_data(data::AbstractMatrix) # Expects (features, time_steps)
    data_mean = mean(data, dims=2)
    data_std = std(data, dims=2)
    data_std = max.(data_std, Float32(1e-8)) 
    normalized_data = (data .- data_mean) ./ data_std
    return normalized_data, data_mean, data_std
end

function denormalize_data(normalized_data::AbstractArray, data_mean::AbstractArray, data_std::AbstractArray)
    return normalized_data .* data_std .+ data_mean
end

# --- Causal Mask Function ---
"""
    create_causal_mask(seq_len)

Create a causal attention mask of size (seq_len, seq_len)
that prevents attending to future positions.
Elements are 0.0f0 for allowed connections and -Inf for disallowed ones.
"""
function create_causal_mask(seq_len::Int)
    # Create row indices and column indices for broadcasting
    row_indices = reshape(1:seq_len, seq_len, 1)
    col_indices = reshape(1:seq_len, 1, seq_len)
    
    # Create the mask: if row_idx >= col_idx, it's an allowed connection (0.0), else -Inf
    mask = ifelse.(row_indices .>= col_indices, 0.0f0, Float32(-Inf))
    return mask # Shape: (seq_len, seq_len)
end


# --- Transformer Model Components ---

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
        dropout = Dropout(Float32(dropout_rate))
        
        new(num_heads, actual_head_dim, W_q, W_k, W_v, W_o, dropout, scale)
    end
end
Flux.@functor MultiHeadAttention

# MODIFIED MultiHeadAttention forward pass to include optional mask
function (mha::MultiHeadAttention)(x::AbstractArray; mask::Union{AbstractMatrix, Nothing}=nothing) # x is (d_model, seq_len, batch_size)
    d_model, seq_len, batch_size = size(x)
    total_dim = mha.num_heads * mha.head_dim

    x_flat = reshape(x, d_model, seq_len * batch_size)

    q_proj_flat = mha.W_q(x_flat)
    k_proj_flat = mha.W_k(x_flat)
    v_proj_flat = mha.W_v(x_flat)

    q_proj = reshape(q_proj_flat, total_dim, seq_len, batch_size)
    k_proj = reshape(k_proj_flat, total_dim, seq_len, batch_size)
    v_proj = reshape(v_proj_flat, total_dim, seq_len, batch_size)

    # Use functional approach instead of mutable collectors
    batch_outputs = map(1:batch_size) do b
        q_batch_seq = view(q_proj, :, :, b) 
        k_batch_seq = view(k_proj, :, :, b) 
        v_batch_seq = view(v_proj, :, :, b) 

        # Process all heads functionally
        head_outputs = map(1:mha.num_heads) do head
            head_start = (head - 1) * mha.head_dim + 1
            head_end = head * mha.head_dim

            q_head = view(q_batch_seq, head_start:head_end, :) 
            k_head = view(k_batch_seq, head_start:head_end, :) 
            v_head = view(v_batch_seq, head_start:head_end, :) 
            
            scores = (q_head' * k_head) .* mha.scale # (seq_len, seq_len)
            
            # Apply mask if provided
            if mask !== nothing
                scores = scores .+ mask 
            end
            
            attention_weights = softmax(scores, dims=2) 
            attention_weights = mha.dropout(attention_weights)

            attended_v = v_head * attention_weights' 
            return attended_v
        end
        
        # Concatenate head outputs vertically
        return vcat(head_outputs...)
    end

    # Concatenate batch outputs along third dimension
    attention_output = cat(batch_outputs...; dims=3) 
    
    attention_output_flat = reshape(attention_output, total_dim, seq_len * batch_size)
    output_proj_flat = mha.W_o(attention_output_flat)
    
    return reshape(output_proj_flat, d_model, seq_len, batch_size)
end


struct PositionalEncoding
    embedding::Matrix{Float32}
    
    function PositionalEncoding(max_len::Int, d_model::Int)
        pe = zeros(Float32, d_model, max_len)
        position = reshape(collect(1:max_len), 1, :) 
        
        for i in 0:(d_model-1)
            denominator = Float32(10000.0^((2 * (i ÷ 2)) / d_model)) 
            angle = position ./ denominator
            if i % 2 == 0
                pe[i+1, :] = sin.(angle)
            else
                pe[i+1, :] = cos.(angle)
            end
        end
        new(pe)
    end
end
Flux.@functor PositionalEncoding 

function (pe::PositionalEncoding)(x) 
    d_model, seq_len, batch_size = size(x)
    max_pe_len = size(pe.embedding, 2)
    len_to_add = min(seq_len, max_pe_len)

    pos_enc_slice = reshape(view(pe.embedding, :, 1:len_to_add), d_model, len_to_add, 1)
    
    if seq_len <= max_pe_len
        return x .+ pos_enc_slice
    else
        output = similar(x)
        output[:, 1:len_to_add, :] = x[:, 1:len_to_add, :] .+ pos_enc_slice
        output[:, (len_to_add+1):end, :] = x[:, (len_to_add+1):end, :] 
        return output
    end
end


struct FeedForward
    layers::Vector{Any} 
    dropout_ff::Dropout 
    
    function FeedForward(d_model::Int, hidden_dims::Vector{Int}, dropout_rate::Float64=0.1)
        ff_layers = Any[]
        current_dim = d_model
        if !isempty(hidden_dims)
            for (i, h_dim) in enumerate(hidden_dims)
                push!(ff_layers, Dense(current_dim, h_dim, relu))
                current_dim = h_dim
            end
        else
            push!(ff_layers, Dense(d_model, d_model)) 
        end
        
        dropout_ff_obj = Dropout(Float32(dropout_rate))
        new(ff_layers, dropout_ff_obj) 
    end
end
Flux.@functor FeedForward (layers,) 

function (ff::FeedForward)(x::AbstractArray) 
    input_feature_dim, seq_len, batch_size = size(x)
    
    x_flat = reshape(x, input_feature_dim, seq_len * batch_size)

    h = x_flat
    for (i, layer) in enumerate(ff.layers)
        h = layer(h)
        if i < length(ff.layers) 
            h = ff.dropout_ff(h) 
        end
    end
    
    last_dense_layer = ff.layers[end] 
    output_dim_of_ff_block = size(last_dense_layer.bias, 1)

    return reshape(h, output_dim_of_ff_block, seq_len, batch_size)
end


struct Transformer
    embedding::Dense
    pos_encoding::PositionalEncoding
    attention::MultiHeadAttention
    norm1::LayerNorm
    feedforward::FeedForward
    output_projection::Dense
    use_causal_mask::Bool # New field to control mask usage
    
    function Transformer(input_dim::Int, d_model::Int, num_heads::Int, hidden_dims_ffn::Vector{Int}, 
                         max_len::Int, dropout_rate::Float64=0.1, head_dim::Int=0; use_causal_mask_flag::Bool=true) # Added flag
        embedding_layer = Dense(input_dim, d_model)
        pos_encoding_layer = PositionalEncoding(max_len, d_model)
        attention_layer = MultiHeadAttention(d_model, num_heads, head_dim, dropout_rate)
        norm1_layer = LayerNorm(d_model)
        feedforward_layer = FeedForward(d_model, hidden_dims_ffn, dropout_rate)
        output_projection_layer = Dense(hidden_dims_ffn[end], input_dim)

        new(embedding_layer, pos_encoding_layer, attention_layer, norm1_layer, 
            feedforward_layer, output_projection_layer, use_causal_mask_flag) # Store flag
    end
end
Flux.@functor Transformer

# MODIFIED Transformer forward pass to create and use causal mask
function (layer::Transformer)(x::AbstractArray) 
    input_feature_dim, seq_len, batch_size = size(x)
    
    x_flat_embed = reshape(x, input_feature_dim, seq_len * batch_size)
    x_embedded_flat = layer.embedding(x_flat_embed)
    d_model = size(x_embedded_flat, 1) 
    x_embedded = reshape(x_embedded_flat, d_model, seq_len, batch_size)
    
    x_pos = layer.pos_encoding(x_embedded)
    
    # Create and pass causal mask if enabled
    local_mask = nothing
    if layer.use_causal_mask
        local_mask = create_causal_mask(seq_len)
        # The mask needs to be compatible with batch processing.
        # If create_causal_mask returns (seq_len, seq_len), it will be broadcasted
        # by Flux across batches and heads correctly when added to scores.
    end
    attended = layer.attention(x_pos; mask=local_mask) # Pass mask here
    
    x1_residual = x_pos .+ attended
    x1_norm = layer.norm1(x1_residual) 
    
    ff_out_seq = layer.feedforward(x1_norm) 
    
    ff_out_last_step = ff_out_seq[:, end, :] 
    
    output_predicted = layer.output_projection(ff_out_last_step) 

    return reshape(output_predicted, input_feature_dim, 1, batch_size)
end


# --- Data Splitting and Batching (Unchanged) ---
function split_data_for_prediction(data::AbstractMatrix, n_inputs::Int, n_pred::Int, train_ratio::Float64=0.8)
    input_dim, n_steps = size(data)
    
    min_sequence_length = n_inputs + n_pred
    if n_steps < min_sequence_length
        error("Data length ($n_steps) is less than required sequence length ($min_sequence_length)")
    end
    
    n_sequences = n_steps - min_sequence_length + 1
    n_train = Int(floor(train_ratio * n_sequences))
    n_val = n_sequences - n_train
    
    train_inputs = zeros(Float32, input_dim, n_inputs, n_train)
    train_targets = zeros(Float32, input_dim, n_pred, n_train)
    val_inputs = zeros(Float32, input_dim, n_inputs, n_val)
    val_targets = zeros(Float32, input_dim, n_pred, n_val)
    
    for i in 1:n_train
        input_start = i
        input_end = i + n_inputs - 1
        target_start = input_end + 1
        target_end = target_start + n_pred - 1
        
        train_inputs[:, :, i] = data[:, input_start:input_end]
        train_targets[:, :, i] = data[:, target_start:target_end]
    end
    
    for i in 1:n_val
        seq_idx = n_train + i
        input_start = seq_idx
        input_end = seq_idx + n_inputs - 1
        target_start = input_end + 1
        target_end = target_start + n_pred - 1
        
        val_inputs[:, :, i] = data[:, input_start:input_end]
        val_targets[:, :, i] = data[:, target_start:target_end]
    end
    
    return train_inputs, train_targets, val_inputs, val_targets
end

function create_training_batches(inputs::AbstractArray, targets::AbstractArray, batch_size::Int)
    num_samples = size(inputs, 3)
    
    actual_batch_size = min(batch_size, num_samples)
    if actual_batch_size == 0 return (similar(inputs, (size(inputs,1), size(inputs,2),0)), similar(targets, (size(targets,1),size(targets,2),0))) end

    indices = randperm(num_samples)[1:actual_batch_size]
    
    batch_inputs = inputs[:, :, indices]
    batch_targets = targets[:, :, indices]
    
    return batch_inputs, batch_targets
end


# --- Main Script Logic (Adjusted for clarity, using your hyperparameters) ---
function run_transformer_training()
    # Model hyperparameters
    input_dim_feat = 1     
    d_model = 32
    hidden_dims_pred_ffn = [128, 32] 
    num_heads = 8
    dropout_rate = 0.1
    head_dim_mha = Int(d_model / num_heads) 
    use_causal_mask_in_transformer = true # Control causal mask usage

    # Training setup
    n_inputs_seq = 32  
    n_pred_steps = 1   
    learning_rate = 1e-4
    n_epochs = 400
    train_batch_size = 200 

    Random.seed!(42)
    println("Generating Lorenz data...")
    lorenz_data_full = generate_lorenz63_data(20000; tspan=(0.0, 200.0)) 
    
    y_data_raw = lorenz_data_full[:, 2:2]' 

    y_data_normalized, y_mean, y_std = normalize_data(y_data_raw)
    println("Data normalization:")
    println("  Original range (y-component): [$(minimum(y_data_raw)), $(maximum(y_data_raw))]")
    println("  Normalized range: [$(minimum(y_data_normalized)), $(maximum(y_data_normalized))]")
    println("  Mean: $(y_mean[1]), Std: $(y_std[1])")

    train_inputs, train_targets, val_inputs, val_targets = split_data_for_prediction(
        y_data_normalized, n_inputs_seq, n_pred_steps, 0.8
    )
    println("Training data shapes: Inputs: ", size(train_inputs), ", Targets: ", size(train_targets))
    println("Validation data shapes: Inputs: ", size(val_inputs), ", Targets: ", size(val_targets))

    if size(train_inputs,3) == 0 || size(val_inputs,3) == 0
        println("Not enough data to create training/validation sets. Exiting.")
        return
    end

    transformer_model = Transformer(input_dim_feat, d_model, num_heads, hidden_dims_pred_ffn, 
                                   n_inputs_seq, dropout_rate, head_dim_mha; 
                                   use_causal_mask_flag=use_causal_mask_in_transformer) # Pass flag

    opt = Adam(learning_rate)
    model_params = Flux.params(transformer_model)

    train_losses = Float32[]
    val_losses = Float32[]
    loss_epochs_tracked = Int[]

    println("\nStarting training...")
    for epoch in 1:n_epochs
        batch_train_inputs, batch_train_targets = create_training_batches(train_inputs, train_targets, train_batch_size)
        if size(batch_train_inputs,3) == 0 continue end

        loss_val, grads_val = Flux.withgradient(model_params) do
            predictions = transformer_model(batch_train_inputs)
            Flux.mse(predictions, batch_train_targets)
        end
        
        Flux.update!(opt, model_params, grads_val)
        
        if epoch % 10 == 0 || epoch == 1
            batch_val_inputs, batch_val_targets = create_training_batches(val_inputs, val_targets, min(32, size(val_inputs, 3)))
            if size(batch_val_inputs,3) == 0 
                val_loss_val = Float32(NaN) 
            else
                val_predictions = transformer_model(batch_val_inputs)
                val_loss_val = Flux.mse(val_predictions, batch_val_targets)
            end

            push!(train_losses, loss_val)
            push!(val_losses, val_loss_val)
            push!(loss_epochs_tracked, epoch)
            
            println("Epoch $epoch: Train Loss = $(round(loss_val, digits=6)), Val Loss = $(round(val_loss_val, digits=6))")
        end
    end

    println("\nTraining complete.")

    if !isempty(loss_epochs_tracked)
        p = plot(loss_epochs_tracked, train_losses, label="Training Loss", xlabel="Epoch", ylabel="MSE Loss", legend=:topright)
        plot!(p, loss_epochs_tracked, val_losses, label="Validation Loss")
        display(p) 
        println("Plot displayed. If running in a non-interactive environment, save the plot to a file.")
    else
        println("No losses tracked to plot.")
    end
end

# To run the training:
run_transformer_training()
