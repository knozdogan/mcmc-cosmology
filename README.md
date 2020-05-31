# mcmc-cosmology
MCMC application for cosmology with Julia

## Introduction
Random walk Metropolis Hastings algorithm is implemented to sample parameters' posterior of cosmological models. This implementation was tested with standard
cosmology model (ΛCDM), which is also implemented in this code (*see ΛCDM() function in mcmc.jl*).

## Usage
A new model can be defined by using *model_name(d::Dict)* and *model=model_name*.
For example:

```julia
@. ΛCDM(s::Dict) = s["H_0"] * sqrt(s["Ω_m"] * (1. + data_frame.z) ^ 3 + 1. - s["Ω_m"]);
model = ΛCDM;
```

Also you can define a statistical model by using this structure:

```julia
prior = product_distribution(...);    # prior distribution

function log_likelihood(s::Dict)
    μ = model(s)
    gauss = MvNormal(μ, PDiagMat(abs2.(σ)))
    return logpdf(gauss,*your_data_arr*)
end

function log_posterior(s::Dict)
    log_prior = logpdf(prior,collect(values(s)));
    if isinf(log_prior)
        return -Inf
    else
        return log_likelihood(s) + log_prior
    end
end
```

Model parameters must follow this structure *Dict{String,Tuple{Float64,Float64}}()*.

```julia
params = Dict{String,Tuple{Float64,Float64}}();
params["H_0"]=(60,75)     # (min,max)
params["Ω_m"]=(0.1,0.6)
```

Also, you have to define initial values for *RandomWalkMetropolisHastings* function:

```julia
init_val = Dict{String, Float64}("H_0"=>65.0,"Ω_m"=>0.1)
```
