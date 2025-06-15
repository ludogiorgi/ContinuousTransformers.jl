"""
Test data generation functionality
"""

@testset "Data Generation Tests" begin
    
    @testset "Lorenz 63 Data Generation" begin
        # Test basic data generation
        data = generate_lorenz63_data(100, tspan=(0.0, 10.0))
        
        @test size(data, 1) == 100
        @test size(data, 2) == 3  # x, y, z coordinates
        @test all(isfinite.(data))  # No NaN or Inf values
        
        # Test with different parameters
        data_large = generate_lorenz63_data(1000, tspan=(0.0, 100.0))
        @test size(data_large, 1) == 1000
        @test size(data_large, 2) == 3
        
        # Test that different parameters produce different results
        data1 = generate_lorenz63_data(50, σ=10.0)
        data2 = generate_lorenz63_data(50, σ=10.1)
        @test data1 != data2
        
        # Test bounds are reasonable for Lorenz system
        x_vals = data[:, 1]
        y_vals = data[:, 2]
        z_vals = data[:, 3]
        
        @test minimum(x_vals) > -30 && maximum(x_vals) < 30
        @test minimum(y_vals) > -30 && maximum(y_vals) < 30
        @test minimum(z_vals) > -5 && maximum(z_vals) < 50
    end
    
    @testset "Data Properties" begin
        # Generate test data
        data = generate_lorenz63_data(1000, tspan=(0.0, 100.0))
        y_data = data[:, 2]
        
        # Test statistical properties
        @test length(y_data) == 1000
        @test std(y_data) > 0  # Should have variation
        @test abs(mean(y_data)) < 5  # Mean should be around 0 for Lorenz
        
        # Test that data is reasonably continuous (allowing for chaotic behavior)
        diffs = diff(y_data)
        @test maximum(abs.(diffs)) < 50  # Relaxed continuity for chaotic system
    end
end
