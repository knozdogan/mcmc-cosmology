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
sample = Dict("H_0"=>65.0,"Ω_m"=>0.1)    # initial value


# define ΛCDM model
@. ΛCDM(s::Dict) = s["H_0"] * sqrt(s["Ω_m"] * (1. + data_frame.z) ^ 3 + 1. - s["Ω_m"]);
model = ΛCDM;
println("Model created")

# prior, log likelihood, and log posterior
params_name = collect(keys(dict));
params_values = collect(values(dict));

dists = [Uniform(val[1],val[2]) for val in params_values];
prior = product_distribution(dists);

σ = data_frame.Herr;

function log_likelihood(mdl,s)
    μ = mdl(s);
    gauss = MvNormal(μ,σ)
    return logpdf(gauss,data_frame.H)
end # function

log_posterior(s::Dict) = log_likelihood(model,s) + logpdf(prior,collect(values(s)));

# test
println(log_likelihood(model,sample))
println(log_posterior(sample))

# define Metropolis-Hasting algorithm
# sample must be a dict
