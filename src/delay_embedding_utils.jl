"""
    DelayEmbeddingProcessor

Processor for creating delay embeddings from time series data with optional normalization.
Supports both univariate (Vector) and multivariate (Matrix) time series.
"""

struct DelayEmbeddingProcessor
    original_data::Matrix{Float32}     # Shape: (n_features, n_timesteps)
    normalized_data::Matrix{Float32}   # Shape: (n_features, n_timesteps)
    embedding_dim::Int
    embedded_data::Matrix{Float32}     # Shape: (n_samples, n_features * embedding_dim)
    # Normalization parameters
    data_mean::Vector{Float32}         # Per-feature means
    data_std::Vector{Float32}          # Per-feature standard deviations
    normalize::Bool
    n_features::Int
end

# Define constructor as a separate function to ensure proper type handling
function DelayEmbeddingProcessor(data::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}, embedding_dim::Int; normalize::Bool=true)
    # Convert input to 2D matrix format: (n_features, n_timesteps)
    if data isa AbstractVector
        data_matrix = reshape(Float32.(data), 1, :)  # Shape: (1, n_timesteps)
        n_features = 1
    else
        # Handle Matrix input - check if it's (n_timesteps, n_features) format
        if size(data, 2) == 1 || size(data, 1) > size(data, 2)
            # Likely (n_timesteps, n_features) format, transpose to (n_features, n_timesteps)
            data_matrix = Float32.(data')  
        else
            # Already (n_features, n_timesteps) format
            data_matrix = Float32.(data)   
        end
        n_features = size(data_matrix, 1)
    end
    
    # Compute normalization parameters per feature
    if normalize
        data_mean = vec(Statistics.mean(data_matrix, dims=2))  
        data_std = vec(Statistics.std(data_matrix, dims=2))    
        
        # Avoid division by zero
        for i in 1:length(data_std)
            if data_std[i] < 1e-8
                @warn "Feature $i has very small standard deviation ($(data_std[i])), using std=1.0"
                data_std[i] = 1.0f0
            end
        end
        
        # Normalize each feature
        normalized_data = (data_matrix .- data_mean) ./ data_std
    else
        data_mean = zeros(Float32, n_features)
        data_std = ones(Float32, n_features)
        normalized_data = copy(data_matrix)
    end
    
    # Create delay embedding from normalized data
    embedded = create_delay_embedding(normalized_data, embedding_dim)
    
    return DelayEmbeddingProcessor(data_matrix, normalized_data, embedding_dim, embedded, data_mean, data_std, normalize, n_features)
end

"""
    create_delay_embedding(data, embedding_dim)

Create delay embedding matrix from multivariate time series.
Input: data matrix of shape (n_features, n_timesteps)
Returns: embedded matrix of shape (n_samples, n_features * embedding_dim)
where each row is [f1(t), f2(t), ..., fn(t), f1(t-1), f2(t-1), ..., fn(t-1), ..., f1(t-embedding_dim+1), ..., fn(t-embedding_dim+1)]
"""
function create_delay_embedding(data::Matrix{Float32}, embedding_dim::Int)
    n_features, n_timesteps = size(data)
    
    if n_timesteps < embedding_dim
        error("Data length ($n_timesteps) must be at least embedding dimension ($embedding_dim)")
    end
    
    n_samples = n_timesteps - embedding_dim + 1
    embedded = Matrix{Float32}(undef, n_samples, n_features * embedding_dim)
    
    for i in 1:n_samples
        for j in 1:embedding_dim
            for f in 1:n_features
                # Column index in embedded matrix
                col_idx = (j-1) * n_features + f
                # Time index in original data (newest to oldest)
                time_idx = i + j - 1
                embedded[i, col_idx] = data[f, time_idx]
            end
        end
    end
    
    return embedded
end

"""
    get_training_sequences(processor, sequence_length)

Generate training sequences from delay-embedded data.
Returns (inputs, targets) where inputs are sequences of embedded vectors
and targets are the next values in the normalized time series.
"""
function get_training_sequences(processor::DelayEmbeddingProcessor, sequence_length::Int)
    embedded_data = processor.embedded_data
    normalized_data = processor.normalized_data  # Use normalized data for targets
    embedding_dim = processor.embedding_dim
    
    n_embedded = size(embedded_data, 1)
    n_sequences = n_embedded - sequence_length
    
    if n_sequences <= 0
        error("Not enough data for sequences of length $sequence_length")
    end
    
    # Input sequences: (sequence_length, embedding_dim, n_sequences)
    inputs = Array{Float32}(undef, sequence_length, embedding_dim, n_sequences)
    # Target values: (1, n_sequences) - predicting next value
    targets = Array{Float32}(undef, 1, n_sequences)
    
    for i in 1:n_sequences
        # Input sequence from embedded data - fix the dimension assignment
        for t in 1:sequence_length
            for d in 1:embedding_dim
                inputs[t, d, i] = embedded_data[i + t - 1, d]
            end
        end
        
        # Target is the next value in normalized time series
        # The target corresponds to the next point after the last embedding vector
        target_idx = i + sequence_length + embedding_dim - 1
        if target_idx <= length(normalized_data)
            targets[1, i] = normalized_data[target_idx]
        else
            targets[1, i] = normalized_data[end]  # Use last available value
        end
    end
    
    return inputs, targets
end

"""
    normalize_value(processor, value)

Normalize a single value using the processor's normalization parameters.
"""
function normalize_value(processor::DelayEmbeddingProcessor, value::Real)
    if !processor.normalize
        return Float32(value)
    end
    return Float32((value - processor.data_mean) / processor.data_std)
end

"""
    denormalize_predictions(processor, predictions)

Convert normalized predictions back to original scale.
"""
function denormalize_predictions(processor::DelayEmbeddingProcessor, predictions::Vector{<:Real})
    if !processor.normalize
        return Float32.(predictions)
    end
    return Float32.([denormalize_value(processor, pred) for pred in predictions])
end

"""
    get_normalization_info(processor)

Get normalization parameters for the processor.
"""
function get_normalization_info(processor::DelayEmbeddingProcessor)
    return (
        normalized=processor.normalize,
        mean=processor.data_mean,
        std=processor.data_std,
        original_range=(minimum(processor.original_data), maximum(processor.original_data)),
        normalized_range=processor.normalize ? (minimum(processor.normalized_data), maximum(processor.normalized_data)) : nothing
    )
end

"""
    denormalize_value(processor, normalized_value)

Convert a normalized value back to original scale using the processor's normalization parameters.
"""
function denormalize_value(processor::DelayEmbeddingProcessor, normalized_value::Float32)
    if processor.normalize
        # Handle multivariate case - use first feature's parameters for now
        return normalized_value * processor.data_std[1] + processor.data_mean[1]
    else
        return normalized_value
    end
end

"""
    get_embedding_data(processor, dim=1)

Get delay embedding data and targets for transformer training.
Returns (inputs, targets) where inputs is (embedding_features, n_samples) and targets is (n_features, n_samples).
"""
function get_embedding_data(processor::DelayEmbeddingProcessor, dim::Int=1)
    normalized_data = processor.normalized_data  # Shape: (n_features, n_timesteps)
    embedding_dim = processor.embedding_dim
    n_features = processor.n_features
    n_timesteps = size(normalized_data, 2)
    
    # We need at least embedding_dim + 1 points to create one input-target pair
    n_samples = n_timesteps - embedding_dim
    if n_samples <= 0
        error("Not enough data for embedding dimension $embedding_dim")
    end
    
    # Input matrix: (n_features * embedding_dim, n_samples)
    input_features = n_features * embedding_dim
    inputs = Array{Float32}(undef, input_features, n_samples)
    # Target matrix: (n_features, n_samples)
    targets = Array{Float32}(undef, n_features, n_samples)
    
    for i in 1:n_samples
        # Create delay embedding for sample i
        for j in 1:embedding_dim
            for f in 1:n_features
                # Input feature index
                input_idx = (j-1) * n_features + f
                # Time index (newest to oldest)
                time_idx = i + j - 1
                inputs[input_idx, i] = normalized_data[f, time_idx]
            end
        end
        
        # Target: next values for all features
        target_time_idx = i + embedding_dim
        for f in 1:n_features
            targets[f, i] = normalized_data[f, target_time_idx]
        end
    end
    
    return inputs, targets
end