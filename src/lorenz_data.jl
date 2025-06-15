"""
    lorenz63!(du, u, p, t)

Lorenz 63 system differential equation.
"""
function lorenz63!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

"""
    generate_lorenz63_data(n_points::Int; tspan=(0.0, 100.0), dt=nothing, σ=10.0, ρ=28.0, β=8/3, return_dt=false)

Generate time series data from the Lorenz 63 system.

# Arguments
- `n_points::Int`: Number of data points to generate
- `tspan`: Time span as (start_time, end_time)
- `dt`: Time step (if nothing, automatically calculated)
- `σ`, `ρ`, `β`: Lorenz system parameters
- `return_dt`: If true, return (data, dt) tuple instead of just data

# Returns
- If return_dt=false: Matrix of size (n_points, 3) containing [x, y, z] coordinates
- If return_dt=true: Tuple (data, dt) where data is the matrix and dt is the time step
"""
function generate_lorenz63_data(n_points::Int; tspan=(0.0, 100.0), dt=nothing, σ=10.0, ρ=28.0, β=8/3, return_dt=false)
    # Initial conditions
    u0 = [1.0, 1.0, 1.0]
    
    # Parameters
    p = [σ, ρ, β]
    
    # Calculate time step if not provided
    if dt === nothing
        dt = (tspan[2] - tspan[1]) / (n_points - 1)
    end
    
    # Create time vector
    t = range(tspan[1], tspan[2], length=n_points)
    
    # Define the ODE problem
    prob = ODEProblem(lorenz63!, u0, tspan, p)
    
    # Solve the ODE
    sol = solve(prob, Tsit5(), saveat=t, reltol=1e-8, abstol=1e-8)
    
    # Convert to matrix format
    data = hcat(sol.u...)'  # Transpose to get (n_points, 3)
    data_f32 = Float32.(data)
    
    if return_dt
        return data_f32, Float64(dt)
    else
        return data_f32
    end
end

"""
    generate_lorenz63_data(n_points::Int, tspan::Tuple; kwargs...)

Convenience method for generate_lorenz63_data with positional tspan argument.
"""
function generate_lorenz63_data(n_points::Int, tspan::Tuple; kwargs...)
    return generate_lorenz63_data(n_points; tspan=tspan, kwargs...)
end