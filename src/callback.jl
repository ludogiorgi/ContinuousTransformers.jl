"""
    combined_prediction_analysis(model, val_inputs_full, val_targets_full; 
                                n_ens=50, seq_len=32, n_preds_example=100, 
                                max_n_preds=150, n_pred_steps=15, seed=42, dt=0.05)

Combined analysis function that generates ensemble trajectory plots and RMSE scaling analysis.

# Arguments
- `model`: Trained ContinuousTransformerModel
- `val_inputs_full`: Validation input data 
- `val_targets_full`: Validation target data
- `n_ens`: Number of ensemble trajectories to generate (default: 50)
- `seq_len`: Sequence length for input windows (default: 32)
- `n_preds_example`: Number of prediction steps for example trajectories (default: 100)
- `max_n_preds`: Maximum prediction horizon for RMSE analysis (default: 150)
- `n_pred_steps`: Number of different prediction horizons to test for RMSE (default: 15)
- `seed`: Random seed for reproducible results (default: 42)
- `dt`: Time step between consecutive data points (default: 0.05)

# Returns
- `combined_plot`: Combined plot with trajectory examples (top 2 rows) and RMSE scaling (bottom row)
- `ensemble_predictions`: Prediction trajectories for examples
- `ensemble_observations`: Observation trajectories for examples  
- `n_pred_horizons`: Prediction horizon values tested for RMSE
- `rmse_scaling`: RMSE values for each horizon
"""
function combined_prediction_analysis(model::ContinuousTransformerModel, 
                                    val_inputs_full::AbstractArray, 
                                    val_targets_full::AbstractArray; 
                                    n_ens::Int=50, seq_len::Int=32, n_preds_example::Int=100, 
                                    max_n_preds::Int=150, n_pred_steps::Int=15, seed=42, dt::Float64=0.05)
    
    println("\n=== Combined Prediction Analysis ===")
    
    # Generate ensemble predictions for example trajectories
    println("Generating ensemble predictions for example trajectories...")
    ensemble_predictions, ensemble_observations, ensemble_indices = generate_ensemble_predictions(
        model, val_inputs_full, val_targets_full; 
        n_ens=n_ens, seq_len=seq_len, n_preds=n_preds_example, seed=seed
    )
    
    # Analyze RMSE scaling with prediction horizon
    println("Analyzing RMSE scaling with prediction horizon...")
    n_pred_horizons, rmse_scaling, rmse_scaling_std, _ = analyze_prediction_horizon_scaling(
        model, val_inputs_full, val_targets_full;
        n_ens=n_ens, seq_len=seq_len, max_n_preds=max_n_preds, n_pred_steps=n_pred_steps, seed=seed
    )
    
    # Create individual trajectory plots (first 6 trajectories)
    n_plots_to_use = min(6, size(ensemble_predictions, 2))
    pred_time = (1:n_preds_example) .* dt  # Convert steps to time units
    trajectory_plots = []
    
    for plot_idx in 1:n_plots_to_use
        # Show legend only in the first panel
        legend_setting = plot_idx == 1 ? :topright : false
        
        # Create individual trajectory plot
        p = plot(title="Trajectory $plot_idx (End: $(ensemble_indices[plot_idx]))", 
                xlabel="Time", ylabel="Value", legend=legend_setting,
                titlefontsize=10)
        
        # Plot predictions
        plot!(p, pred_time, ensemble_predictions[:, plot_idx], 
              label="Predictions", color=:red, linewidth=2, marker=:circle, markersize=3)
        
        # Plot ground truth
        plot!(p, pred_time, ensemble_observations[:, plot_idx], 
              label="Ground Truth", color=:green, linewidth=2, marker=:square, markersize=3)
        
        push!(trajectory_plots, p)
    end
    
    # Create RMSE scaling plot (spans full width)
    horizon_time = n_pred_horizons .* dt  # Convert horizon steps to time units
    rmse_plot = plot(horizon_time, rmse_scaling, 
                    ribbon=rmse_scaling_std,
                    xlabel="Prediction Horizon (time units)", 
                    ylabel="RMSE",
                    title="Prediction Error Scaling with Horizon",
                    label="RMSE ± σ",
                    linewidth=3,
                    marker=:circle,
                    markersize=5,
                    legend=:topleft,
                    grid=true,
                    titlefontsize=12)
    
    # Create statistical analysis plots
    println("Computing statistical distributions and autocorrelations...")
    
    # Flatten all predictions and observations for statistical analysis
    all_predictions = vec(ensemble_predictions)
    all_observations = vec(ensemble_observations)
    
    # PDF comparison plot
    pdf_plot = plot(title="Probability Density Functions", 
                   xlabel="Value", ylabel="Density",
                   legend=:topright, titlefontsize=12)
    
    # Create histograms normalized to PDFs
    histogram!(pdf_plot, all_observations, bins=50, alpha=0.6, normalize=:pdf,
              label="Observed", color=:green)
    histogram!(pdf_plot, all_predictions, bins=50, alpha=0.6, normalize=:pdf,
              label="Predicted", color=:red)
    
    # ACF comparison plot
    acf_plot = plot(title="Autocorrelation Functions", 
                   xlabel="Lag (time units)", ylabel="Autocorrelation",
                   legend=:topright, titlefontsize=12, grid=true)
    
    # Compute autocorrelation functions for all trajectories
    max_lag = min(50, n_preds_example ÷ 2)  # Reasonable max lag
    lags = (0:max_lag) .* dt
    
    # Initialize arrays to store ACFs for all trajectories
    all_obs_acfs = Array{Float64}(undef, max_lag + 1, size(ensemble_observations, 2))
    all_pred_acfs = Array{Float64}(undef, max_lag + 1, size(ensemble_predictions, 2))
    
    # Compute ACF for each trajectory
    for traj_idx in 1:size(ensemble_observations, 2)
        obs_acf = autocorr(ensemble_observations[:, traj_idx], 0:max_lag)
        pred_acf = autocorr(ensemble_predictions[:, traj_idx], 0:max_lag)
        
        all_obs_acfs[:, traj_idx] = obs_acf
        all_pred_acfs[:, traj_idx] = pred_acf
    end
    
    # Compute average ACFs across all trajectories
    avg_obs_acf = mean(all_obs_acfs, dims=2)[:, 1]
    avg_pred_acf = mean(all_pred_acfs, dims=2)[:, 1]
    
    # Compute standard deviations for error bands
    std_obs_acf = std(all_obs_acfs, dims=2)[:, 1]
    std_pred_acf = std(all_pred_acfs, dims=2)[:, 1]
    
    # Plot average ACFs with error bands
    plot!(acf_plot, lags, avg_obs_acf, ribbon=std_obs_acf, label="Observed (avg ± σ)", 
          color=:green, linewidth=3, marker=:circle, markersize=4, alpha=0.7)
    plot!(acf_plot, lags, avg_pred_acf, ribbon=std_pred_acf, label="Predicted (avg ± σ)", 
          color=:red, linewidth=3, marker=:square, markersize=4, alpha=0.7)
    
    # Add horizontal line at zero for reference
    hline!(acf_plot, [0], color=:black, linestyle=:dash, alpha=0.5, label="")

    # Create combined layout: 2 rows of 3 trajectory plots + 1 row for RMSE + 1 row for statistical analysis
    l = @layout [a{0.33w} b{0.33w} c{0.33w}
                 d{0.33w} e{0.33w} f{0.33w}
                 g{1.0w}
                 h{0.5w} i{0.5w}]
    
    combined_plot = plot(trajectory_plots[1], trajectory_plots[2], trajectory_plots[3],
                        trajectory_plots[4], trajectory_plots[5], trajectory_plots[6],
                        rmse_plot,
                        pdf_plot, acf_plot,
                        layout=l, 
                        size=(1200, 1200),
                        plot_title="Transformer Prediction Analysis")
    
    # Print summary statistics
    println("\n=== Analysis Summary ===")
    println("Example trajectories:")
    println("  - Generated $(size(ensemble_predictions, 2)) ensemble trajectories")
    println("  - Each trajectory has $(size(ensemble_predictions, 1)) prediction steps")
    println("  - Time step (dt): $dt")
    println("  - Total prediction time: $(round(n_preds_example * dt, digits=3))")
    
    println("\nRMSE scaling analysis:")
    println("  - Shortest horizon ($(n_pred_horizons[1]) steps, $(round(n_pred_horizons[1] * dt, digits=3)) time): RMSE = $(round(rmse_scaling[1], digits=4))")
    println("  - Longest horizon ($(n_pred_horizons[end]) steps, $(round(n_pred_horizons[end] * dt, digits=3)) time): RMSE = $(round(rmse_scaling[end], digits=4))")
    println("  - RMSE growth factor: $(round(rmse_scaling[end] / rmse_scaling[1], digits=2))x")
    
    # Calculate average growth rate
    log_growth_rate = (log(rmse_scaling[end]) - log(rmse_scaling[1])) / (n_pred_horizons[end] - n_pred_horizons[1])
    println("  - Exponential growth rate: $(round(log_growth_rate, digits=6)) per step")
    println("  - Exponential growth rate: $(round(log_growth_rate / dt, digits=6)) per time unit")
    
    return combined_plot, ensemble_predictions, ensemble_observations, n_pred_horizons, rmse_scaling
end

"""
    autocorr(x, lags)

Compute autocorrelation function for lags.
"""
function autocorr(x::Vector, lags::AbstractRange)
    n = length(x)
    x_centered = x .- mean(x)
    var_x = var(x, corrected=false)
    
    acf = Float64[]
    for lag in lags
        if lag == 0
            push!(acf, 1.0)
        elseif lag < n
            correlation = sum(x_centered[1:n-lag] .* x_centered[1+lag:n]) / ((n - lag) * var_x)
            push!(acf, correlation)
        else
            push!(acf, 0.0)
        end
    end
    
    return acf
end