"""
    predict_multistep(model, processor, n_preds; verbose=true)

Perform multi-step prediction using the trained transformer model.
Starting point is chosen randomly from the validation data.

# Arguments
- `model`: Trained ContinuousTransformerModel
- `processor`: DelayEmbeddingProcessor used for training
- `n_preds`: Number of prediction steps
- `verbose`: Whether to print progress information

# Returns
- `predicted_values_denormalized`: Vector of predicted values in original scale
- `ground_truth_denormalized`: Vector of ground truth values in original scale  
- `initial_sequence_denormalized`: Initial sequence used for prediction in original scale
- `random_val_start_idx`: Starting index in normalized_data for reproducibility
"""
function predict_multistep(model::ContinuousTransformerModel, processor::DelayEmbeddingProcessor, n_preds::Int; verbose::Bool=true)
    
    # 1. Determine the split point for validation data
    temp_inputs_for_split_calc, _ = get_embedding_data(processor)
    total_embedding_samples = size(temp_inputs_for_split_calc, 2)
    n_train_embedding_samples = Int(round(0.8 * total_embedding_samples))
    
    # Validation data starts after training embeddings
    val_data_start_idx_in_norm_data = n_train_embedding_samples + 1
    
    # 2. Calculate valid range for starting prediction
    max_random_start_for_initial_sequence = length(processor.normalized_data) - processor.embedding_dim - n_preds + 1
    
    if val_data_start_idx_in_norm_data > max_random_start_for_initial_sequence
        error("""
        Validation set is too small to select a starting sequence of length $(processor.embedding_dim) 
        and then predict $n_preds steps.
        Available start for validation embeddings: index $val_data_start_idx_in_norm_data in normalized_data.
        Latest possible start for prediction task: index $max_random_start_for_initial_sequence in normalized_data.
        Consider reducing n_preds, using a smaller embedding_dim, or generating more data.
        Length of normalized_data: $(length(processor.normalized_data))
        """)
    end
    
    # 3. Select random starting point from validation data
    random_val_start_idx = rand(val_data_start_idx_in_norm_data:max_random_start_for_initial_sequence)
    verbose && println("Selected random start index in normalized_data for prediction: $random_val_start_idx")
    
    # 4. Extract initial sequence
    initial_sequence_normalized = processor.normalized_data[random_val_start_idx:random_val_start_idx + processor.embedding_dim - 1]
    current_input_embedding = reshape(Float32.(initial_sequence_normalized), processor.embedding_dim, 1)
    
    # 5. Perform iterative prediction
    predicted_values_normalized = Vector{Float32}(undef, n_preds)
    verbose && println("Starting iterative prediction for $n_preds steps...")
    
    for i in 1:n_preds
        # Get prediction from model (2D input returns (output_dim, seq_len))
        pred_normalized_value = model(current_input_embedding)[1, 1] 
        predicted_values_normalized[i] = pred_normalized_value
        
        # Update embedding for next prediction: shift left and add new prediction
        current_input_embedding = vcat(current_input_embedding[2:end, :], fill(pred_normalized_value, (1, 1)))
    end
    
    verbose && println("Iterative prediction finished.")
    
    # 6. Extract ground truth values
    ground_truth_start_idx_in_norm_data = random_val_start_idx + processor.embedding_dim
    ground_truth_end_idx_in_norm_data = ground_truth_start_idx_in_norm_data + n_preds - 1
    
    if ground_truth_end_idx_in_norm_data > length(processor.normalized_data)
        error("Attempting to access ground truth beyond the bounds of normalized_data.")
    end
    
    ground_truth_normalized = processor.normalized_data[ground_truth_start_idx_in_norm_data:ground_truth_end_idx_in_norm_data]
    
    # 7. Denormalize all values
    predicted_values_denormalized = denormalize_predictions(processor, predicted_values_normalized)
    ground_truth_denormalized = denormalize_predictions(processor, ground_truth_normalized) 
    initial_sequence_denormalized = denormalize_predictions(processor, initial_sequence_normalized)
    
    return predicted_values_denormalized, ground_truth_denormalized, initial_sequence_denormalized, random_val_start_idx
end

"""
    plot_prediction_results(predicted_values, ground_truth, initial_sequence, start_idx, processor; n_show=200)

Plot the results of multi-step prediction.

# Arguments
- `predicted_values`: Predicted values (denormalized)
- `ground_truth`: Ground truth values (denormalized)  
- `initial_sequence`: Initial sequence (denormalized)
- `start_idx`: Starting index for the prediction
- `processor`: DelayEmbeddingProcessor for context
- `n_show`: Number of prediction steps to show in plot
"""
function plot_prediction_results(predicted_values, ground_truth, initial_sequence, start_idx, processor; n_show=200)
    n_preds = min(length(predicted_values), n_show)
    
    # Time axis for the initial input sequence (relative to the first prediction point)
    time_steps_initial = (-processor.embedding_dim+1):0 
    # Time axis for the predicted and ground truth values
    time_steps_pred = 1:n_preds
    
    plot_title = "Multi-step Prediction (Lorenz Y) - Start Index $(start_idx)"
    
    p = plot(
        time_steps_initial, 
        initial_sequence, 
        label="Initial Input (Observed)", 
        color=:grey, 
        linestyle=:dash,
        linewidth=1.5
    )
    
    plot!(p,
        time_steps_pred, 
        ground_truth[1:n_preds], 
        label="Ground Truth (Observed)", 
        color=:blue, 
        linewidth=2
    )
    
    plot!(p,
        time_steps_pred, 
        predicted_values[1:n_preds], 
        label="Transformer Prediction", 
        color=:red, 
        linestyle=:dashdot, 
        linewidth=2
    )
    
    title!(p, plot_title)
    xlabel!(p, "Time Steps (from end of initial sequence)")
    ylabel!(p, "Value (Denormalized)")
    
    return p
end

# Example usage:
predicted, ground_truth, initial_seq, start_idx = predict_multistep(model, processor, 200)
p = plot_prediction_results(predicted, ground_truth, initial_seq, start_idx, processor)
display(p)

##

# Plot validation predictions
println("\n5. Plotting validation predictions...")

# Get the training/validation split
inputs, targets = get_embedding_data(processor)
n_total_samples = size(inputs, 2)
n_train_samples = Int(round(0.8 * n_total_samples))

# Get validation inputs and targets
val_inputs = inputs[:, n_train_samples+1:end]
val_targets = targets[:, n_train_samples+1:end]

# Take first 100 validation samples
n_plot = min(100, size(val_inputs, 2))
val_inputs_plot = val_inputs[:, 1:n_plot]
val_targets_plot = val_targets[:, 1:n_plot]

# Get predictions for validation inputs
val_predictions = model(val_inputs_plot)

# Denormalize for plotting
val_targets_denorm = denormalize_predictions(processor, val_targets_plot[1, :])
val_predictions_denorm = denormalize_predictions(processor, val_predictions[1, :])

mean(abs.(val_targets_plot .- val_predictions))  # Ensure targets are in correct shape for plotting
# Create the plot
p_val = plot(
    1:n_plot,
    val_targets_denorm,
    label="Ground Truth",
    color=:blue,
    linewidth=2,
    title="Validation Set: First $n_plot Values vs Transformer Predictions",
    xlabel="Sample Index",
    ylabel="Value (Denormalized)"
)

plot!(p_val,
    1:n_plot,
    val_predictions_denorm,
    label="Transformer Prediction",
    color=:red,
    linestyle=:dash,
    linewidth=2
)

# Add some statistics
mse_val = sum((val_targets_denorm .- val_predictions_denorm).^2) / n_plot
println("Validation MSE (first $n_plot samples): $(round(mse_val, digits=4))")

display(p_val)