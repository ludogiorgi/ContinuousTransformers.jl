"""
Test TimeSeriesProcessor functionality
"""

using TimeSeriesTransformers
using Statistics

@testset "TimeSeriesProcessor Tests" begin
    
    @testset "Processor Creation" begin
        # Generate test data
        data = generate_lorenz63_data(500, tspan=(0.0, 50.0))
        y_data = data[:, 2]
        
        # Test processor creation with different cluster numbers
        for num_clusters in [5, 8, 10, 15]
            processor = TimeSeriesProcessor(y_data, num_clusters)
            
            # Test that processor is created with correct parameters
            @test processor.num_clusters == num_clusters
            @test length(processor.raw_data) == length(y_data)
            @test length(processor.normalized_data) == length(y_data)
            
            # Run clustering process
            TimeSeriesTransformers.process(processor)
            
            # After processing, cluster centers should be populated
            @test length(processor.cluster_centers) > 0
            @test length(processor.cluster_centers) <= num_clusters  # May be fewer due to uniqueness
            @test all(isfinite.(processor.cluster_centers))
            
            # Test cluster centers are within data range (with some tolerance for normalization)
            data_min, data_max = extrema(y_data)
            @test all(center -> data_min - 3*std(y_data) <= center <= data_max + 3*std(y_data), 
                     processor.cluster_centers)
        end
    end
    
    @testset "Sequence Dataset Creation" begin
        # Generate test data
        data = generate_lorenz63_data(200, tspan=(0.0, 20.0))
        y_data = data[:, 2]
        processor = TimeSeriesProcessor(y_data, 8)
        
        # Test different sequence lengths
        for seq_len in [5, 10, 16, 20]
            X, y = get_sequence_dataset(processor, seq_len)
            
            # X is Vector{Vector{Int}}, each sequence has length seq_len
            @test length(X) == length(y_data) - seq_len  # Number of sequences
            @test length(y) == length(y_data) - seq_len   # Number of targets
            @test all(seq -> length(seq) == seq_len, X)   # Each sequence has correct length
            
            # Test that sequences are valid cluster indices
            @test all(seq -> all(x -> x >= 1, seq), X)    # All elements >= 1
            @test all(seq -> all(x -> x <= 8, seq), X)    # All elements <= num_clusters  
            @test all(y_val -> y_val >= 1, y)            # All targets >= 1
            @test all(y_val -> y_val <= 8, y)            # All targets <= num_clusters
        end
    end
    
    @testset "Data Conversion" begin
        # Simple test case with known data
        simple_data = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5]
        processor = TimeSeriesProcessor(simple_data, 3)
        
        # Run clustering process
        TimeSeriesTransformers.process(processor)
        
        # Test that clustering produces reasonable results
        @test length(processor.cluster_centers) == 3
        
        # Test sequence creation
        X, y = get_sequence_dataset(processor, 3)
        @test length(X) == length(simple_data) - 3     # Number of sequences
        @test length(y) == length(simple_data) - 3     # Number of targets
        @test all(seq -> length(seq) == 3, X)          # Each sequence has length 3
    end
    
    @testset "Edge Cases" begin
        # Test with minimal data
        minimal_data = [1.0, 2.0, 3.0]
        processor = TimeSeriesProcessor(minimal_data, 2)
        
        # Run clustering process
        TimeSeriesTransformers.process(processor)
        
        @test length(processor.cluster_centers) == 2
        
        # Test very short sequences
        X, y = get_sequence_dataset(processor, 1)
        @test length(X) == 2                          # Number of sequences: 3 - 1 = 2
        @test length(y) == 2                          # Number of targets
        @test all(seq -> length(seq) == 1, X)         # Each sequence has length 1
    end
end
