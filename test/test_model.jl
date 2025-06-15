"""
Test TransformerModel functionality  
"""

using Test
using TimeSeriesTransformers
using Flux

@testset "TransformerModel Tests" begin
    
    @testset "Model Creation" begin
        # Test basic model creation
        model = TransformerModel(
            num_clusters = 8,
            latent_dim = 8,
            d_model = 32,
            num_heads = 4,
            num_layers = 2,
            dropout_rate = 0.1
        )
        
        @test model isa TransformerModel
        
        # Test parameter counting
        params = Flux.params(model)
        param_count = sum(length, params)
        @test param_count > 0
        
        # Test model configuration validation - should warn, not error
        model_with_warning = TransformerModel(
            num_clusters = 8,
            latent_dim = 8, 
            d_model = 33,  # Not divisible by num_heads - should warn
            num_heads = 4,
            num_layers = 2
        )
        @test model_with_warning isa TransformerModel
    end
    
    @testset "Model Forward Pass" begin
        # Create test model
        num_clusters = 6
        model = TransformerModel(
            num_clusters = num_clusters,
            latent_dim = num_clusters,
            d_model = 24,
            num_heads = 3,
            num_layers = 1,
            dropout_rate = 0.0  # No dropout for testing
        )
        
        # Test forward pass with different input sizes
        batch_size = 4
        seq_length = 8
        
        # Create test input (vector of vectors format)
        input_sequences = [rand(1:num_clusters, seq_length) for _ in 1:batch_size]
        
        # Forward pass
        output = model(input_sequences)
        
        @test size(output) == (num_clusters, batch_size)
        @test all(isfinite.(output))
        
        # Test with different batch sizes
        for bs in [1, 2, 8]
            test_input = [rand(1:num_clusters, seq_length) for _ in 1:bs]
            test_output = model(test_input)
            @test size(test_output) == (num_clusters, bs)
        end
    end
    
    @testset "Model Components" begin
        # Test individual components exist and are callable
        num_clusters = 4
        d_model = 16
        num_heads = 2
        
        # Test that we can create components directly
        embedding = Flux.Embedding(num_clusters, d_model)
        @test embedding isa Flux.Embedding
        
        # Test positional encoding
        seq_len = 10
        max_len = 100
        pos_enc = PositionalEncoding(max_len, d_model)
        test_input = randn(Float32, d_model, seq_len, 2)  # (d_model, seq_len, batch)
        pos_output = pos_enc(test_input)
        @test size(pos_output) == size(test_input)
        
        # Test multi-head attention
        mha = TimeSeriesTransformers.MultiHeadAttention(d_model, num_heads)
        attn_output = mha(test_input)
        @test size(attn_output) == size(test_input)
    end
    
    @testset "Gradient Computation" begin
        # Test that gradients can be computed
        model = TransformerModel(
            num_clusters = 4,
            latent_dim = 4,
            d_model = 16,
            num_heads = 2,
            num_layers = 1
        )
        
        # Test data
        input_sequences = [rand(1:4, 6) for _ in 1:2]  # Vector of vectors format
        target = rand(1:4, 2)
        
        # Define simple loss
        loss_fn = () -> begin
            output = model(input_sequences)
            # Convert targets to one-hot for cross-entropy  
            target_onehot = Flux.onehotbatch(target, 1:4)
            return Flux.logitcrossentropy(output, target_onehot)
        end
        
        # Compute gradients
        grads = Flux.gradient(loss_fn, Flux.params(model))
        
        @test !isempty(grads)
        # Check that some gradients are non-zero
        has_nonzero_grad = any(p -> any(grads[p] .!= 0), Flux.params(model))
        @test has_nonzero_grad
    end
end
