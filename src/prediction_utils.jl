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
function generate_ensemble_predictions(model::ContinuousTransformerModel, 
                                     val_inputs_full::AbstractArray, 
                                     val_targets_full::AbstractArray; 
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
    analyze_prediction_horizon_scaling(model, val_inputs_full, val_targets_full; kwargs...)

Analyze how prediction error scales with prediction horizon.
"""
function analyze_prediction_horizon_scaling(model::ContinuousTransformerModel, 
                                          val_inputs_full::AbstractArray, 
                                          val_targets_full::AbstractArray;
                                          n_ens::Int=50, seq_len::Int=32, max_n_preds::Int=150, 
                                          n_pred_steps::Int=15, seed=42)
    
    # Set random seed if specified
    if seed !== nothing
        Random.seed!(seed)
    end
    
    # Create prediction horizons to test
    n_pred_horizons = round.(Int, range(10, max_n_preds, length=n_pred_steps))
    rmse_scaling = Float32[]
    rmse_scaling_std = Float32[]
    
    for n_preds in n_pred_horizons
        println("Testing prediction horizon: $n_preds steps")
        
        # Generate ensemble predictions for this horizon
        predictions, observations, _ = generate_ensemble_predictions(
            model, val_inputs_full, val_targets_full; 
            n_ens=n_ens, seq_len=seq_len, n_preds=n_preds, seed=seed
        )
        
        # Calculate RMSE for each ensemble member
        ensemble_rmses = Float32[]
        for ens_idx in 1:size(predictions, 2)
            rmse = sqrt(mean((predictions[:, ens_idx] .- observations[:, ens_idx]).^2))
            push!(ensemble_rmses, rmse)
        end
        
        # Store mean and std of RMSE across ensemble
        push!(rmse_scaling, mean(ensemble_rmses))
        push!(rmse_scaling_std, std(ensemble_rmses))
    end
    
    return n_pred_horizons, rmse_scaling, rmse_scaling_std, nothing
end

using Statistics
using Plots

"""
    predict_next_values(model, input_sequence, n_steps=1)

Predict next values using the trained model.
"""
function predict_next_values(model, input_sequence::AbstractArray, n_steps::Int=1)
    # Ensure input is 3D: (features, seq_len, batch_size)
    if ndims(input_sequence) == 2
        input_sequence = reshape(input_sequence, size(input_sequence)..., 1)
    end
    
    predictions = model(input_sequence)  # Shape: (output_dim, seq_len, batch_size, n_ode_steps)
    
    # Return final time step predictions
    return predictions[:, :, :, end]
end

"""
    combined_prediction_analysis(model, val_inputs, val_targets; kwargs...)

Combined prediction analysis with ensemble predictions and horizon scaling.
"""
function combined_prediction_analysis(
    model, val_inputs, val_targets;
    n_ens=100, seq_len=32, n_preds_example=50, max_n_preds=100, n_pred_steps=10, seed=42, dt=0.01
)
    Random.seed!(seed)
    
    # Generate ensemble predictions
    ensemble_preds, ensemble_obs, _ = generate_ensemble_predictions(
        model, val_inputs, val_targets, n_ens, seq_len, n_preds_example
    )
    
    # Analyze prediction horizon scaling
    horizons, rmse_values = analyze_prediction_horizon_scaling(
        model, val_inputs, val_targets, max_n_preds, n_pred_steps, seq_len
    )
    
    # Create combined plot
    p1 = plot(ensemble_obs[1:min(200, end)], label="Observed", color=:black, linewidth=2)
    plot!(p1, ensemble_preds[1:min(200, end)], label="Predicted", color=:red, linewidth=1, alpha=0.7)
    xlabel!(p1, "Time Step")
    ylabel!(p1, "Value")
    title!(p1, "Ensemble Prediction Example")
    
    p2 = plot(horizons .* dt, rmse_values, marker=:circle, linewidth=2)
    xlabel!(p2, "Prediction Horizon (time units)")
    ylabel!(p2, "RMSE")
    title!(p2, "Prediction Horizon Scaling")
    
    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    return combined_plot, ensemble_preds, ensemble_obs, horizons, rmse_values
end

"""
    generate_ensemble_predictions(model, val_inputs, val_targets, n_ens, seq_len, n_preds)

Generate ensemble predictions for analysis.
"""
function generate_ensemble_predictions(model, val_inputs, val_targets, n_ens, seq_len, n_preds)
    n_val_samples = size(val_inputs, 2)
    
    # Randomly select starting points
    start_indices = rand(1:(n_val_samples - seq_len - n_preds), n_ens)
    
    all_preds = Float32[]
    all_obs = Float32[]
    
    for start_idx in start_indices
        # Get input sequence
        input_seq = val_inputs[:, start_idx:(start_idx + seq_len - 1)]
        input_3d = reshape(input_seq, size(input_seq)..., 1)
        
        # Get observed values
        obs_vals = val_targets[1, (start_idx + seq_len):(start_idx + seq_len + n_preds - 1)]
        
        # Generate predictions
        predictions = model(input_3d)
        pred_vals = predictions[1, :, 1, end]  # Use final time step
        
        append!(all_preds, pred_vals[1:min(n_preds, length(pred_vals))])
        append!(all_obs, obs_vals[1:min(n_preds, length(obs_vals))])
    end
    
    return all_preds, all_obs, start_indices
end

"""
    analyze_prediction_horizon_scaling(model, val_inputs, val_targets, max_n_preds, n_pred_steps, seq_len)

Analyze how prediction error scales with horizon.
"""
function analyze_prediction_horizon_scaling(model, val_inputs, val_targets, max_n_preds, n_pred_steps, seq_len)
    horizons = 1:n_pred_steps:max_n_preds
    rmse_values = Float32[]
    
    for horizon in horizons
        # Generate predictions for this horizon
        preds, obs, _ = generate_ensemble_predictions(
            model, val_inputs, val_targets, 50, seq_len, horizon
        )
        
        # Calculate RMSE
        if length(preds) > 0 && length(obs) > 0
            min_len = min(length(preds), length(obs))
            rmse = sqrt(mean((preds[1:min_len] .- obs[1:min_len]).^2))
            push!(rmse_values, rmse)
        else
            push!(rmse_values, NaN32)
        end
    end
    
    return horizons, rmse_values
end

"""
    autocorr(x, lags)

Compute autocorrelation function.
"""
function autocorr(x::Vector, lags::AbstractVector{<:Integer})
    n = length(x)
    x_centered = x .- mean(x)
    autocorrs = Float64[]
    
    for lag in lags
        if lag == 0
            push!(autocorrs, 1.0)
        elseif lag < n
            c = sum(x_centered[1:(n-lag)] .* x_centered[(1+lag):n]) / (n - lag)
            c0 = sum(x_centered.^2) / n
            push!(autocorrs, c / c0)
        else
            push!(autocorrs, 0.0)
        end
    end
    
    return autocorrs
end