using ABDA
using Statistics
using PyPlot
close("all")


## create the Bayesian model from scratch using discrete parameter (page 19--22) 

# prior: θ is descrete (this function is not needed but included for calrity) 
function log_prior(θ)
    if θ ∈ 1.0:4.0
        return log(0.25)
    else
        return -Inf
    end
end

# The likeligood
function log_likelihood(θ,y)
    n = length(y)
    σ = 1.16 # Strange std to use
    ε = (y .- θ)./σ
    return -n*0.5*log(2π) - n*log(σ) - 0.5*ε'*ε  
end

# our data given on page 20 
y = [1.77, 2.23, 2.70]   


# log posterior
function log_posterior(θ) 
    rθ = round(θ[1]) # here we use the trick of rounding it
    return log_likelihood(rθ,y) + log_prior(rθ)
end


θ_init = [1.0]
θs, lps = ABDA.slice_sample(copy(θ_init), 100*ones(length(θ_init)), log_posterior, 10_000)
rθs = round.(θs)

N_burn_in = 500
# remove "burn in" phase
rθs = rθs[:,N_burn_in+1:end]
lps = lps[N_burn_in+1:end] 

# plot mcmc chain
fr = [sum(rθs[1,:] .== k)/length(rθs[1,:]) for k = 1:4] 
figure()
subplot(211), plot(rθs[1,:],".",alpha=0.5)
xlabel(raw"index $i$")
ylabel(raw"$\theta_i \sim p(\theta|y)$")

subplot(212), bar(1:4,fr)
xlabel(raw"$\theta$")
ylabel(raw"$p(\theta|y)$")
tight_layout()

println("fr: ", fr)
