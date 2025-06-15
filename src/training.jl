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