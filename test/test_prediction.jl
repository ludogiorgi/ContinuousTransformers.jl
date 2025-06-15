"""
Test prediction and generation functionality
"""

using Test
using TimeSeriesTransformers
using Flux

@testset "Prediction and Generation Tests" begin
    
    @testset "Basic Prediction" begin
        # Setup test data and model
        data = generate_lorenz63_data(100, tspan=(0.0, 10.0))
        y_data = data[:, 2]
        processor = TimeSeriesProcessor(y_data, 4)
        
        model = TransformerModel(
            num_clusters = 4,
            latent_dim = 4,
            d_model = 16,
            num_heads = 2,
            num_layers = 1
        )
        
        # Test single prediction
        test_sequence = rand(1:4, 6)  # Random sequence of cluster indices
        
        # Test predict function exists and returns reasonable output
        if isdefined(TimeSeriesTransformers, :predict)
            prediction = predict(model, test_sequence)
            @test prediction isa Integer
            @test 1 <= prediction <= 4
        end
        
        # Test basic model probability output
        output = model([test_sequence])
        probs = Flux.softmax(output, dims=1)[:, 1]
        @test length(probs) == 4  # num_clusters
        @test all(probs .>= 0)
        @test sum(probs) ≈ 1.0 atol=1e-6  # Probabilities should sum to 1
    end
    
    @testset "Sequence Generation" begin
        # Setup
        data = generate_lorenz63_data(80, tspan=(0.0, 8.0))
        y_data = data[:, 2]
        processor = TimeSeriesProcessor(y_data, 5)
        
        model = TransformerModel(
            num_clusters = 5,
            latent_dim = 5,
            d_model = 20,
            num_heads = 4,
            num_layers = 1
        )
        
        # Test generation if function exists
        if isdefined(TimeSeriesTransformers, :generate_time_series)
            seed_sequence = rand(1:5, 8)
            generation_length = 20
            
            generated_clusters, generated_values = generate_time_series(
                model, processor, seed_sequence, generation_length
            )
            
            @test length(generated_clusters) == generation_length
            @test length(generated_values) == generation_length
            @test all(1 .<= generated_clusters .<= 5)
            @test all(isfinite.(generated_values))
        end
    end
    
    @testset "Validation Accuracy" begin
        # Test accuracy computation
        data = generate_lorenz63_data(60, tspan=(0.0, 6.0))
        y_data = data[:, 2]
        processor = TimeSeriesProcessor(y_data, 3)
        
        model = TransformerModel(
            num_clusters = 3,
            latent_dim = 3,
            d_model = 12,
            num_heads = 3,
            num_layers = 1
        )
        
        # Create test dataset
        X, y = get_sequence_dataset(processor, 5)
        
        # Compute validation accuracy
        accuracy_result = compute_validation_accuracy(model, X, y, verbose=false)
        accuracy = accuracy_result[1]  # Extract accuracy from tuple
        predictions = accuracy_result[2]  # Extract predictions
        
        @test 0 <= accuracy <= 1  # Accuracy should be between 0 and 1
        @test accuracy isa Number
        @test isfinite(accuracy)
        @test length(predictions) == length(y)  # Should have prediction for each target
    end
    
    @testset "Model Output Properties" begin
        # Test that model outputs have expected properties
        model = TransformerModel(
            num_clusters = 6,
            latent_dim = 6,
            d_model = 18,
            num_heads = 3,
            num_layers = 1
        )
        
        test_input = [rand(1:6, 10) for _ in 1:3]  # Vector of vectors format
        output = model(test_input)
        
        @test size(output) == (6, 3)  # (num_clusters, batch_size)
        @test all(isfinite.(output))
        
        # Test that softmax gives valid probabilities
        probs = Flux.softmax(output, dims=1)
        @test all(probs .>= 0)
        @test all(sum(probs, dims=1) .≈ 1.0)
    end
end
