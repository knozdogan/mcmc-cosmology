using StatsPlots
using StatsBase
using LinearAlgebra
using JLD

samples = load("./samples_ΛCDM.jld")["samples"]
println("> samples loaded")
plt_title = "ΛCDM"


function matrix_plot(data::Dict, plt_title::String, num_points=10000, num_bins=100)

    println("> matrix-plotting ...")

    param_names = collect(keys(data))
    num_params = length(param_names)
    num_samples = length(samples[param_names[1]])

    idx = rand(1:num_samples, num_points)
    plots = Any[]
    for i in 1:num_params, j in 1:num_params
        if i==j
            # density
            d = density(samples[param_names[i]],label=param_names[i],yaxis=false,
                legend=:none)
            push!(plots,d)
        end
        if i>j
            # scatter
            s = scatter(samples[param_names[j]][idx],samples[param_names[i]][idx],
                m = (0.5, [:cross], 1), xlabel=param_names[j],
                ylabel=param_names[i],legend=:none)
            push!(plots,s)
        end
        if i<j
            # histogram 2d
            h = fit(Histogram, (samples[param_names[j]],samples[param_names[i]]),nbins=num_bins)
            norm_h = normalize(h, mode=:probability)
            hist = plot(norm_h,show_empty_bins=true,legend=:none,xlabel=param_names[j],
                   ylabel=param_names[i])
            push!(plots, hist)
        end
    end
    plot(plots..., layout=(num_params,num_params),plot_title=plt_title,framestyle=:box)
    savefig("./model_$(plt_title).png")
end # function

# test
@time matrix_plot(samples,plt_title)
