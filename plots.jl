using Plots,
    StatsBase,
    StatsPlots,
    LinearAlgebra,
    JLD,
    LaTeXStrings

import PyPlot
pyplot();

samples = load("./samples_ΛCDM.jld")["samples"]
println("> samples loaded")
plt_title = "ΛCDM"


function corner_plot(data::Dict, plt_title::String, num_points=10000, num_bins=100)

    println("> corner-plotting ...")

    param_names = collect(keys(data))
    num_params = length(param_names)
    num_samples = length(samples[param_names[1]])
    latex_format = latexstring.(param_names)

    idx = rand(1:num_samples, num_points)
    plots = Any[]
    for i in 1:num_params, j in 1:num_params
        if i==j
            # density
            d = density(samples[param_names[i]],showaxis=:x,label=latex_format[i],
                framestyle=:box,linecolor=:black,lw=3)
                # label=...
            push!(plots,d)
        end
        # if i>j
        #     # scatter
        #     s = scatter(samples[param_names[j]][idx],samples[param_names[i]][idx],
        #         m = (0.5, [:cross], 1),legend=:none)
        #     push!(plots,s)
        # end
        if i<j
            blank = plot(showaxis = false, grid = false, legend=:none)
            push!(plots,blank)
        end
        if i>j
            # histogram 2d
            h = fit(Histogram, (samples[param_names[j]],samples[param_names[i]]),nbins=num_bins)
            norm_h = normalize(h, mode=:probability)
            hist = plot(norm_h,show_empty_bins=true,legend=:none,
                xlabel=latex_format[j], ylabel=latex_format[i])
            push!(plots, hist)
        end
    end
    # plot_title: not currently implemented. see plots attributes on
    # https://docs.juliaplots.org/latest/generated/attributes_plot/

    # layout design
    # l = @layout [
    #     a{0.1w} [grid(num_params,num_params)
    #             b{0.1h}  ]]
    plot(plots..., layout=(num_params,num_params))
    savefig("./model_$(plt_title).png")
end # function

# test
@time corner_plot(samples,plt_title)
