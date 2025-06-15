"""
    DelayEmbeddingProcessor

Processor for creating delay embeddings from time series data with optional normalization.
"""

struct DelayEmbeddingProcessor
    original_data::Vector{Float32}
    normalized_data::Vector{Float32}
    embedding_dim::Int
    embedded_data::Matrix{Float32}
    # Normalization parameters
    data_mean::Float32
    data_std::Float32
    normalize::Bool
    
    function DelayEmbeddingProcessor(data::Vector{<:Real}, embedding_dim::Int; normalize::Bool=true)
        data_f32 = Float32.(data)
        
        # Compute normalization parameters
        if normalize
            data_mean = Float32(Statistics.mean(data_f32))
            data_std = Float32(Statistics.std(data_f32))
            # Avoid division by zero
            if data_std < 1e-8
                @warn "Data has very small standard deviation ($data_std), using std=1.0"
                data_std = 1.0f0
            end
            normalized_data = (data_f32 .- data_mean) ./ data_std
        else
            data_mean = 0.0f0
            data_std = 1.0f0
            normalized_data = copy(data_f32)
        end
        
        # Create delay embedding from normalized data
        embedded = create_delay_embedding(normalized_data, embedding_dim)
        new(data_f32, normalized_data, embedding_dim, embedded, data_mean, data_std, normalize)
    end
end

"""
    create_delay_embedding(data, embedding_dim)

Create delay embedding matrix from 1D time series.
Returns matrix of size (length(data) - embedding_dim + 1, embedding_dim)
where each row is [y(t), y(t-1), ..., y(t-embedding_dim+1)]
"""
function create_delay_embedding(data::Vector{Float32}, embedding_dim::Int)
    n = length(data)
    if n < embedding_dim
        error("Data length ($n) must be at least embedding dimension ($embedding_dim)")
    end
    
    embedded_length = n - embedding_dim + 1
    embedded = Matrix{Float32}(undef, embedded_length, embedding_dim)
    
    for i in 1:embedded_length
        for j in 1:embedding_dim
            embedded[i, j] = data[i + j - 1]
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
        # Use the correct field names from DelayEmbeddingProcessor struct
        return normalized_value * processor.data_std + processor.data_mean
    else
        return normalized_value
    end
end

"""
    get_embedding_data(processor, dim=1)

Get delay embedding data and targets for transformer training.
Returns (inputs, targets) where inputs is (embedding_dim, n_samples) and targets is (dim, n_samples).
The delay embedding is structured so that inputs[end, N] = targets[1, N-1].
"""
function get_embedding_data(processor::DelayEmbeddingProcessor, dim::Int=1)
    normalized_data = processor.normalized_data
    embedding_dim = processor.embedding_dim
    
    # We need at least embedding_dim + 1 points to create one input-target pair
    n_samples = length(normalized_data) - embedding_dim
    if n_samples <= 0
        error("Not enough data for embedding dimension $embedding_dim")
    end
    
    # Input matrix: (embedding_dim, n_samples)
    inputs = Array{Float32}(undef, embedding_dim, n_samples)
    # Target matrix: (1, n_samples)
    targets = Array{Float32}(undef, 1, n_samples)
    
    for i in 1:n_samples
        # Create delay embedding: [y_{i+embedding_dim-1}, y_{i+embedding_dim-2}, ..., y_i]
        # This means inputs[1, i] = y_{i+embedding_dim-1} (newest)
        # and inputs[end, i] = y_i (oldest)
        for j in 1:embedding_dim
            inputs[j, i] = normalized_data[i + j - 1]
        end
        # Target: y_{i+embedding_dim} (the next value after the embedding window)
        targets[1, i] = normalized_data[i + embedding_dim]
    end
    
    return inputs, targets
end