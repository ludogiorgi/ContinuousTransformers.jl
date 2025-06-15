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