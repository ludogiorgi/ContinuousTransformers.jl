using DifferentialEquations

"""
    lorenz63!(du, u, p, t)

Lorenz 63 system differential equation.
"""
function lorenz63!(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
end

"""
    generate_lorenz63_data(n_points; tspan=(0.0, 100.0), u0=[1.0, 1.0, 1.0], p=[10.0, 28.0, 8/3], return_dt=false)

Generate Lorenz 63 time series data.
"""
function generate_lorenz63_data(n_points::Int; tspan=(0.0, 100.0), u0=[1.0, 1.0, 1.0], p=[10.0, 28.0, 8/3], return_dt=false)
    prob = ODEProblem(lorenz63!, u0, tspan, p)
    
    # Calculate time points
    t_eval = range(tspan[1], tspan[2], length=n_points)
    dt = (tspan[2] - tspan[1]) / (n_points - 1)
    
    sol = solve(prob, Tsit5(), saveat=t_eval)
    
    # Convert to matrix form
    data = hcat(sol.u...)'  # Shape: (n_points, 3)
    
    if return_dt
        return Float32.(data), Float32(dt)
    else
        return Float32.(data)
    end
end