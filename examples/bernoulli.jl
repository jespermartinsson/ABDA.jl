using ABDA
using PyPlot

# data
y = [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]

# log-likelihood bernoulli
function log_likelihood(θ::Float64,y::Union{Vector{Float64},Vector{Int64}})
    if (0<θ<1)
        n = length(y)
        llh = 0.0
        for i in 1:n
            llh += y[i]*log(θ) + (1-y[i])*log(1-θ)
        end
        return llh
    else
        return -Inf
    end
    
end

# log prior
function log_beta(x,α,β)
    if 0<x<1
        return (α-1)*log(x) + (β-1)*log(1-x)
    else
        return -Inf
    end
end

# log posterior
log_posterior(θ) = log_likelihood(θ[1],y) + log_beta(θ[1],1,1)

θ_init = [0.5]
## sample the posterior
N_burn_in  = 500
θ_samp, lps = sample(copy(θ_init), ones(length(θ_init)), log_posterior, 10_000, N_burn_in)


# remove "burn in" phase
θ_samp = θ_samp[:,N_burn_in+1:end]
lps = lps[N_burn_in+1:end] 



# plot mcmc chain
figure()
subplot(1,2,1), plot(θ_samp[1,:])
subplot(1,2,2), hist(θ_samp[1,:],1000)
tight_layout()