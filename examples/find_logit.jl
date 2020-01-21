using ABDA
using PyPlot
using Statistics
using Random

using PyPlot
close("all")
# kriberg/lastbil44/d97c8839
include("set_backtrack_data.jl")

# add delay
pop!(fb_dir)
pushfirst!(fb_dir,1)

for n = 1:0
    pushfirst!(fb_dir,fb_dir[1])
    pushfirst!(is_bt,is_bt[1])
    pushfirst!(is_ft,is_ft[1])
    pushfirst!(est_dir,est_dir[1])
end


N = length(est_dir)


x1 = is_bt
x2 = is_ft
x3 = (est_dir .== 1).*1

y = (fb_dir .== 1 ).*1

X = hcat(ones(N), x1, x2, x3, x1.*x2, x1.*x3, x2.*x3, x1.*x2.*x3)
beta = randn(size(X,2))



function loglike(beta,X,y)
    y_hat = X*beta .> 0
    e = y .- y_hat
    return e'*e
end

function logistic(x)
    return 1/(1 + exp(-x))
end

# log-likelihood bernoulli
function log_likelihood(beta,X,y)
    θ = logistic.(X*beta) 
    n = length(y)
    llh = 0.0
    for i in 1:n
        llh += y[i]*log(θ[i]) + (1-y[i])*log(1-θ[i])
    end
    return llh
end

function log_prior(beta)
    return -beta'*beta*0.01
end

println(loglike(beta,X,y))
println(log_likelihood(beta,X,y))

using Optim

a = optimize((beta) -> log_likelihood(beta,X,y), beta, NelderMead())
beta_est = a.minimizer
θ_hat = logistic.(X*beta_est) 
figure()
plot(y,"ko-")
plot(θ_hat,"ro-")



## sample the posterior
beta_init = randn(size(X,2))
N_burn_in  = 50000
beta_samp, lps = slice_sample(copy(beta_init), ones(length(beta_init)).*10, (beta) -> log_likelihood(beta,X,y) + log_prior(beta), 100_000; printing=true)

# remove "burn in" phase
beta_samp = beta_samp[:,N_burn_in+1:end]
lps = lps[N_burn_in+1:end] 



# plot mcmc chain
figure()
subplot(1,2,1), plot(beta_samp[1,:])
subplot(1,2,2), hist(beta_samp[1,:],1000)
tight_layout()

ind = argmax(lps)
# beta_hat = median(beta_samp,dims=2)
beta_hat = beta_samp[:,ind]
θ_hat = logistic.(X*beta_hat) 
figure()
plot(y,"ko-")
plot(θ_hat .> 0.5 ,"ro-")




