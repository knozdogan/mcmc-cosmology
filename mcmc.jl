using DelimitedFiles
using DataFrames
using Distributions
using LinearAlgebra
using PDMats
using JLD
# using Traceur

cd(pwd());

# load data and create dataframe
data,header = readdlm("./data/zH_chen_2017.dat",',';header=true);
data_frame = DataFrame(z=data[:,1],H=data[:,2],Herr=data[:,3]);
println("> Data loaded")

# define parameters
params = Dict{String,Tuple{Float64,Float64}}();
params["H_0"]=(60,75)     # (min,max)
params["Ω_m"]=(0.1,0.6)
init_val = Dict{String, Float64}("H_0"=>65.0,"Ω_m"=>0.1)    # initial value
params_values = collect(values(params))

######################
# Cosmological model #
######################
@. ΛCDM(s::Dict) = s["H_0"] * sqrt(s["Ω_m"] * (1. + data_frame.z) ^ 3 + 1. - s["Ω_m"]);
model = ΛCDM;
println("> Cosmological model $(model) created")

#####################
# Statistical model #
#####################
dists = [Uniform(val[1],val[2]) for val in params_values];
prior = product_distribution(dists);    # prior distribution

σ = data_frame.Herr;

function log_likelihood(s::Dict)
    μ = model(s)
    gauss = MvNormal(μ, PDiagMat(abs2.(σ)))
    return logpdf(gauss,data_frame.H)
end

function log_posterior(s::Dict)
    log_prior = logpdf(prior,collect(values(s)));
    if isinf(log_prior)
        return -Inf
    else
        return log_likelihood(s) + log_prior
    end
end

println("> Statistical model defined")

############################################
# Random Walk Metropolis-Hasting algorithm #
############################################
"""
    RandomWalkMetropolisHastings(ln_posterior, init_vals, steps=1e6, burn_in=1e3, step_size=1)

"""
function RandomWalkMetropolisHastings(ln_posterior, init_vals::Dict, steps=1e7, burn_in=1e4, step_size=0.05)
    params_names = collect(keys(init_vals));
    θ_init = collect(values(init_vals))
    num_params = length(params_names);
    samples = Dict{String, Vector{Float64}}();

    key_names = copy(params_names)
    push!(key_names, "$(ln_posterior)")

    step_var = step_size * ones(Float64,num_params);    # variance
    Q(μ) = MvNormal(μ,step_var);   # proposal distribution

    overall_steps = steps+burn_in;
    for i in 1:overall_steps
        progress_bar(i,overall_steps);

        θ_cand = rand(Q(θ_init))
        cand_dict = Dict{String, Float64}(params_names[i]=>θ_cand[i] for i in 1:num_params);
        init_dict = Dict{String, Float64}(params_names[i]=>θ_init[i] for i in 1:num_params);

        log_post = ln_posterior(cand_dict)
        log_likelihood_ratio = log_post - ln_posterior(init_dict)


        log_α = min(0, log_likelihood_ratio);
        u = log(rand(Uniform(0,1)));

        if u < log_α
            if i>burn_in
                data = copy(θ_cand)
                push!(data, log_post)
                add_arr_to_dict(samples, data, key_names)
            end
            θ_init = θ_cand;
        # else
        #     if i>burn_in
        #         add_sample(samples,θ_init,params_names)
        #     end
        end # acceptance procedure
    end
    println("> RW-MH..")
    println(">> Acceptance Rate: $(length(samples[params_names[1]])/steps)")
    return samples
end

# push sample to array in dict
function add_arr_to_dict(dict::Dict,sample::Array,key_names::Array)
    num_params = length(key_names)
    for i in 1:num_params
        vals = get!(Vector{Float64}, dict, key_names[i])
        push!(vals, sample[i])
    end
end # function

function progress_bar(i,steps,update=100)
    if i%update==0
        progress = i*100/steps
        print("> Running... $(round(Int,progress))%   \r")
    end
end


# test
@time chains = RandomWalkMetropolisHastings(log_posterior,init_val);
save("./samples/samples_$(model)_log.jld", "samples", chains)
println("> Saved!")
