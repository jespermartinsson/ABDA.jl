using ABDA
using Statistics
using PyPlot
close("all")



# linear model
function model(β,x)
    X = hcat(ones(length(x)),x) 
    return X*β
end


## create a toy regression example
x = collect(0:0.01:1)
n = length(x)

# true parameters
β0 = [1.0, 2.0]
σ0 = 0.5

# linear model
X = hcat(ones(n),x) 
y0 = model(β0,x)

# add normally distributed noise e ~ N(0,simga0)
e = σ0*randn(n)
y = y0 .+ e

# plot result
figure()
plot(x,y,"k.")
plot(x,y0,"r-")





## create the Bayesian model from scratch

# initial paramteters 
β_init = [1.0, 1.0]
σ_init = 1.0
θ_init = vcat(β_init, σ_init)


# log-likelihood
function log_likelihood(θ,y,x)
    β = θ[1:end-1]
    σ = θ[end]
    n = length(y)
    if σ <=0 
        return -Inf
    else
        ε = (y .- model(β,x))./σ
        return -n*0.5*log(2π) - n*log(σ) - 0.5*ε'*ε  
    end
end
    
# log-prior
function log_prior(θ)
    β = θ[1:end-1]
    σ = θ[end]
    return log_prior_beta(β) + log_prior_sigma(σ)    
end

function log_prior_beta(β)
    return 0.0
end

function log_prior_sigma(σ)
    if σ <=0 
        return -Inf
    else
        return 0.0#1/σ
    end
end

# log posterior
log_posterior(θ) = log_likelihood(θ,y,x) + log_prior(θ)




## sample the posterior
N_burn_in  = 500
θs, lps = sample(copy(θ_init), ones(length(θ_init)), log_posterior, 10_000, N_burn_in)

# remove "burn in" phase
θs= θs[:,N_burn_in+1:end]
lps = lps[N_burn_in+1:end] 

# plot mcmc chain
figure()
subplot(321), plot(θs[1,:])
subplot(322), hist(θs[1,:],1000)
subplot(323), plot(θs[2,:])
subplot(324), hist(θs[2,:],1000)
subplot(325), plot(θs[3,:])
subplot(326), hist(θs[3,:],1000)




# plot estimate
θ = mean(θs,dims=2)
β = θ[1:end-1]
σ = θ[end]
y_est = model(β,x)

figure()
plot(x,y,"k.")

# plot creadible lines
n = size(θs,2)
for i in 1:100:n
    βi = θs[1:end-1,i]
    σi = θs[end,i]
    y_est_i = model(βi,x)
    plot(x,y_est_i,"b-",alpha=0.05)
end

# plot estimate
plot(x,y_est,"b-",lw=2,alpha=0.5)
plot(x,y0,"r-",lw=2,alpha=1.0)
