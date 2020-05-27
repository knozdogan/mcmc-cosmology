using DelimitedFiles
using DataFrames
using Distributions

# load data and create dataframe
cd("/home/kaan/JuliaProjects/mcmc-cosmology");
data,header = readdlm("./zH_chen_2017.dat",',';header=true);
data_frame = DataFrame(z=P[:,1],H=P[:,2],Herr=P[:,3])

# define ΛCDM model


# define log prior, log likelihood, and log posterior


# define Metropolis-Hasting algorithm
