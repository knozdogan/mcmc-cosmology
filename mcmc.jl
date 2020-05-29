using DelimitedFiles
using DataFrames
using Distributions

# load data and create dataframe
cd(pwd());
data,header = readdlm("./zH_chen_2017.dat",',';header=true);
data_frame = DataFrame(z=data[:,1],H=data[:,2],Herr=data[:,3]);

# define parameters
dict = Dict{String,Tuple{Float64,Float64}}();
dict["H_0"]=(50,80)     # (min,max)
dict["Ω_m"]=(0,0.6)


# define ΛCDM model
@. ΛCDM(s::Dict) = s["H_0"] * sqrt(s["Ω_m"] * (1. + data_frame.z) ^ 3 + 1. - s["Ω_m"]);
println("Model created")

# prior, log likelihood, and log posterior
params_name = collect(keys(dict));
params_values = collect(values(dict));

dists = [Uniform(val[1],val[2]) for val in params_values];
prior = product_distribution(dists);

function log_likelihood(s::Dict)
    # Distributions can be used to define log_likelihood
    σ2 = data_frame.Herr .^ 2;
    model = ΛCDM(s);
    residu = (data_frame.H .- model) .^ 2 ./ σ2 .+ log.(2 .* pi .* σ2);
    return -0.5 * (sum(residu,dims=1)[1])
end # log_likelihood

log_posterior(s::Dict) = log_likelihood(s) + logpdf(prior,collect(values(s)));

# test
sample = Dict("H_0"=>65.0,"Ω_m"=>0.1);
println(log_likelihood(sample))
println(log_posterior(sample))

# define Metropolis-Hasting algorithm
# sample must be a dict
