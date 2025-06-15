"""
Test training functionality (minimal tests for speed)
"""

using Test
using TimeSeriesTransformers

@testset "Training Tests" begin
    
    @testset "Basic Training Setup" begin
        # Generate minimal test data
        data = generate_lorenz63_data(100, tspan=(0.0, 10.0))
        y_data = data[:, 2]
        processor = TimeSeriesProcessor(y_data, 4)
        
        # Create small model for fast testing
        model = TransformerModel(
            num_clusters = 4,
            latent_dim = 4,
            d_model = 16,
            num_heads = 2,
            num_layers = 1,
            dropout_rate = 0.1
        )
        
        # Test that model and data are compatible
        X, y = get_sequence_dataset(processor, 5)
        @test length(X) > 0  # Have sequences
        @test all(length(seq) == 5 for seq in X)  # All sequences have correct length
        @test all(all(1 .<= seq .<= 4) for seq in X)  # Valid cluster indices
        @test all(y .>= 1) && all(y .<= 4)
    end
    
    @testset "Training Pipeline" begin
        # Generate test data
        data = generate_lorenz63_data(150, tspan=(0.0, 15.0))
        y_data = data[:, 2]
        processor = TimeSeriesProcessor(y_data, 5)
        
        # Create model
        model = TransformerModel(
            num_clusters = 5,
            latent_dim = 5,
            d_model = 20,
            num_heads = 4,
            num_layers = 1,
            dropout_rate = 0.0
        )
        
        # Test short training run (just to verify it doesn't crash)
        try
            trained_model, train_losses, val_losses = train_and_evaluate!(
                model, 
                processor, 
                8;  # sequence_length
                epochs = 2,  # Very short for testing
                batch_size = 4,
                learning_rate = 1f-3,
                early_stopping_patience = 5,
                verbose = false,
                display_plots = false,
                n_batches_per_epoch = 5  # Minimal batches
            )
            
            @test trained_model isa TransformerModel
            @test length(train_losses) <= 2  # May stop early
            @test length(val_losses) <= 2
            @test all(train_losses .>= 0)  # Losses should be non-negative
            @test all(val_losses .>= 0)
            
        catch e
            # If training fails, we still want to know what the error was
            @warn "Training test failed with error: $e"
            rethrow(e)
        end
    end
    
    @testset "Loss Function" begin
        # Test loss computation directly
        model = TransformerModel(
            num_clusters = 3,
            latent_dim = 3,
            d_model = 12,
            num_heads = 3,
            num_layers = 1
        )
        
        # Create test batch (vector of vectors format)
        input_sequences = [rand(1:3, 4) for _ in 1:2]  # 2 sequences of length 4
        targets = rand(1:3, 2)       # (batch_size,)
        
        # Compute loss
        loss = transformer_loss(model, input_sequences, targets)
        
        @test loss isa Number
        @test loss >= 0
        @test isfinite(loss)
    end
end
