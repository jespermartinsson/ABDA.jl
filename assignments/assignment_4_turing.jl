using ABDA
using PyPlot
close("all")

# data
y = [1,0,1,1,0,1,1,1,0,1,1,1,1,1]
z = [1,0,0,0,0,0,0,1,1,0]

# Load Turing and MCMCChains.
using Turing, MCMCChains

# Load the distributions library.
using Distributions

# Load StatsPlots for density plots.
using StatsPlots

@model coinflip(y) = begin
    
    # Our prior belief about the probability of heads in a coin.
    p ~ Beta(1, 1)
    
    # The number of observations.
    N = length(y)
    for n in 1:N
        # Heads or tails of a coin are drawn from a Bernoulli distribution.
        y[n] ~ Bernoulli(p)
    end
end;

# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
iterations = 1000
ϵ = 0.05
τ = 10

# Start sampling.
chain = sample(coinflip(y), NUTS(1500, 200, 0.65));


# Construct summary of the sampling process for the parameter p, i.e. the probability of heads in a coin.
p_summary = chain[:p]
plot(p_summary, seriestype = :histogram)