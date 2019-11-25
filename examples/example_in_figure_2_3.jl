using ABDA
using Statistics
using PyPlot
close("all")


## create the Bayesian model from scratch using a discrete parameter (page 20--22) 

# prior: θ is descrete
function log_prior(θ)
    pmf = [1.0, 1.0, 1.0, 1.0]/4 # these are the probability mass function
    if θ ∈ 1.0:4.0 # if theta in [1.0, 2.0, 3.0, 4.0]
        return log(pmf[Int(θ)])
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

# Our data given on page 20 
y = [1.77, 2.23, 2.70]   


# The log posterior
function log_posterior(θ) 
    rθ = round(θ[1]) # here we use the trick of rounding it to a discrete parameter (hence the prior will always give log(0.25))
    return log_likelihood(rθ,y) + log_prior(rθ)
end


θ_init = [1.0]
θs, lps = ABDA.slice_sample(copy(θ_init), 100*ones(length(θ_init)), log_posterior, 10_000)
rθs = round.(θs)

N_burn_in = 500
# remove "burn in" phase
rθs = rθs[:,N_burn_in+1:end]
lps = lps[N_burn_in+1:end] 

# estimate posterior probabilities based on the mcmc samples
fr = [sum(rθs[1,:] .== k)/length(rθs[1,:]) for k = 1:4] 
println("esrtimated posterior probabilities: ", fr)


# calculate true probabilities
pr = [exp(log_posterior([k])) for k in 1.0:4.0]
pr = pr./sum(pr)
println("true posterior probabilities: ", pr)

figure()
# plot mcmc chain
subplot(211), plot(rθs[1,1:1000],".-",alpha=0.5)
xlabel(raw"index $i$")
ylabel(raw"$\theta_i \sim p(\theta|y)$")

subplot(212), 
bar(1:4,fr,label=raw"est. $p(\theta|y)$ based on mcmc")
plot(1:4,pr,"ok",label=raw"true $p(\theta|y)$")


x = -1:0.01:6
for k in 1:4
    l = exp.(log_likelihood.(k,x))
    l = l./maximum(l)
    if k == 1
        plot(x,l*fr[k],"b-",alpha = 0.5,label=raw"estimated likelihoods")
        plot(x,l*pr[k],"k--",alpha = 0.5,label=raw"true likelihoods")
    else
        plot(x,l*fr[k],"b-",alpha = 0.5)
        plot(x,l*pr[k],"k--",alpha = 0.5)
    end 
end

legend()
xlabel(raw"$\theta$")
ylabel(raw"$p(\theta|y)$")
tight_layout()

