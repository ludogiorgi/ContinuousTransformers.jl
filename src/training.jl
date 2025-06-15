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
    training_batch_size::Int = 4,
    n_future_steps::Int = 10  # Number of future time steps to predict
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
    batch_targets_alloc = Array{Float32}(undef, output_dim, seq_len, training_batch_size, n_future_steps)
    
    for epoch in 1:epochs
        # Training
        epoch_train_loss = 0f0
        actual_training_steps_this_epoch = 0
        
        if size(train_inputs_full, 2) < seq_len + n_future_steps
            @warn "Not enough training data for sequence length $seq_len + $n_future_steps future steps. Skipping epoch $epoch."
            push!(train_losses, NaN32) 
            push!(val_losses, NaN32)   
            continue
        end

        for step in 1:n_training_steps_per_epoch
            for b_idx in 1:training_batch_size 
                # Ensure we have enough data for both input sequence and future targets
                max_start_idx = size(train_inputs_full, 2) - seq_len - n_future_steps + 1
                start_idx = rand(1:max_start_idx)
                
                # Input sequence
                input_end_idx = start_idx + seq_len - 1
                copyto!(view(batch_inputs_alloc, :, :, b_idx), view(train_inputs_full, :, start_idx:input_end_idx))
                
                # Target trajectory: consecutive future time steps
                for t in 1:n_future_steps
                    # For each time step t, we want the sequence shifted by t positions
                    target_start = start_idx + t
                    target_end = min(target_start + seq_len - 1, size(train_targets_full, 2))
                    
                    if target_end >= target_start
                        seq_length = target_end - target_start + 1
                        copyto!(view(batch_targets_alloc, :, 1:seq_length, b_idx, t), 
                               view(train_targets_full, :, target_start:target_end))
                        # Fill remaining positions with the last available value if needed
                        if seq_length < seq_len
                            batch_targets_alloc[:, (seq_length+1):seq_len, b_idx, t] .= batch_targets_alloc[:, seq_length, b_idx, t]
                        end
                    else
                        # If we run out of data, use the last available value
                        batch_targets_alloc[:, :, b_idx, t] .= train_targets_full[:, end]
                    end
                end
            end
            
            loss, grads = Flux.withgradient(model) do m
                predictions = m(batch_inputs_alloc) # Shape: (output_dim, seq_len, training_batch_size, n_steps)
                
                # Ensure predictions and targets have compatible dimensions
                pred_n_steps = size(predictions, 4)
                target_n_steps = min(pred_n_steps, n_future_steps)
                
                # Use only the matching number of time steps
                pred_subset = predictions[:, :, :, 1:target_n_steps]
                target_subset = batch_targets_alloc[:, :, :, 1:target_n_steps]
                
                Flux.mse(pred_subset, target_subset)
            end
            
            if isnan(loss) || isinf(loss)
                @warn "Training loss is $loss at epoch $epoch, step $step. Skipping update."
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
            if size(val_inputs_full, 2) < val_seq_len + n_future_steps
                 @warn "Not enough validation data for val_seq_len $val_seq_len + $n_future_steps future steps. Processing what's available."
            end

            for val_batch_start_idx in 1:(val_seq_len + n_future_steps):size(val_inputs_full, 2)
                val_input_end_idx = min(val_batch_start_idx + val_seq_len - 1, size(val_inputs_full, 2))
                
                if val_batch_start_idx > val_input_end_idx
                    continue
                end

                current_val_input_chunk = view(val_inputs_full, :, val_batch_start_idx:val_input_end_idx)
                
                # Create validation target trajectory
                actual_val_seq_len = val_input_end_idx - val_batch_start_idx + 1
                val_targets_4d = Array{Float32}(undef, output_dim, actual_val_seq_len, 1, n_future_steps)
                
                for t in 1:n_future_steps
                    target_start = val_batch_start_idx + t
                    target_end = min(target_start + actual_val_seq_len - 1, size(val_targets_full, 2))
                    
                    if target_end >= target_start
                        seq_length = target_end - target_start + 1
                        copyto!(view(val_targets_4d, :, 1:seq_length, 1, t), 
                               view(val_targets_full, :, target_start:target_end))
                        # Fill remaining positions if needed
                        if seq_length < actual_val_seq_len
                            val_targets_4d[:, (seq_length+1):actual_val_seq_len, 1, t] .= val_targets_4d[:, seq_length, 1, t]
                        end
                    else
                        val_targets_4d[:, :, 1, t] .= val_targets_full[:, end]
                    end
                end
                
                # Reshape validation input to 3D for model
                val_input_3d = reshape(current_val_input_chunk, size(current_val_input_chunk, 1), size(current_val_input_chunk, 2), 1)
                
                val_predictions_chunk = model(val_input_3d) # Shape: (output_dim, seq_len, 1, n_steps)
                
                # Ensure compatible dimensions for validation
                pred_n_steps = size(val_predictions_chunk, 4)
                target_n_steps = min(pred_n_steps, n_future_steps)
                
                pred_subset = val_predictions_chunk[:, :, :, 1:target_n_steps]
                target_subset = val_targets_4d[:, :, :, 1:target_n_steps]
                
                loss_chunk = Flux.mse(pred_subset, target_subset)

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