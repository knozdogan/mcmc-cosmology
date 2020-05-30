using DelimitedFiles
using DataFrames
using Distributions
using JLD

cd(pwd());

# load data and create dataframe
data,header = readdlm("./zH_chen_2017.dat",',';header=true);
data_frame = DataFrame(z=data[:,1],H=data[:,2],Herr=data[:,3]);
println("> Data loaded")

# define parameters
params = Dict{String,Tuple{Float64,Float64}}();
params["H_0"]=(50,80)     # (min,max)
params["Ω_m"]=(0.1,0.6)
init_val = Dict("H_0"=>65.0,"Ω_m"=>0.2)    # initial value
params_name = collect(keys(params));
params_values = collect(values(params));

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
    gauss = MvNormal(model(s),σ)
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
function RandomWalkMetropolisHastings(ln_posterior, init_vals::Dict, steps=1e6, burn_in=1e3, step_size=1)
    params_names = collect(keys(init_vals));
    num_params = length(params_names);
    samples = Dict{String, Vector{Float64}}();

    Q(μ) = MvNormal(μ,step_size);   # proposal distribution
    θ_init = rand(Q(collect(values(init_vals))));

    overall_steps = steps+burn_in;
    for i in 1:overall_steps
        progress_bar(i,overall_steps);

        θ_cand = rand(Q(θ_init))
        cand_dict = Dict(params_names[i]=>θ_cand[i] for i in 1:num_params);
        init_dict = Dict(params_names[i]=>θ_init[i] for i in 1:num_params);

        log_likelihood_ratio = logpdf(Q(θ_cand),θ_init) + ln_posterior(cand_dict) -
                               logpdf(Q(θ_init), θ_cand) - ln_posterior(init_dict)

        log_α = min(0, log_likelihood_ratio);
        u = log(rand(Uniform(0,1)));

        if u < log_α
            if i>burn_in
                add_sample(samples,θ_cand, params_names)
            end
            θ_init = θ_cand;
        else
            if i>burn_in
                add_sample(samples,θ_init,params_names)
            end
        end # acceptance procedure
    end
    return samples
end

# push sample to array in dict
function add_sample(dict::Dict,sample::Array,key_names::Array)
    num_params = length(key_names)
    for i in 1:num_params
        vals = get!(Vector{Float64}, dict, key_names[i])
        push!(vals, sample[i])
    end
end # function

function progress_bar(i::Int,steps::Int,update=100)
    progress = i*100/steps
    if i%update==0
        print("> Running... $(round(Int,progress))%   \r")
    end
end


# test
chains = RandomWalkMetropolisHastings(log_posterior,init_val);
save("samples.jld", "samples", chains)
println("> Saved!")
