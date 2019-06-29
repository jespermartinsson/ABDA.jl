using ABDA
using Statistics
using PyPlot
#close("all")




function sample_cat(ϕ,N=1)
    y = Vector{Float64}(undef,N)
    cϕ = cumsum(ϕ)
    for n in 1:N
        u = rand()
        y[n] = 1 + sum(u .>= cϕ)
    end
    if N == 1
        return y[N]
    else
        return y
    end
end

if false
    # test sampling from categorical
    ϕ = [1,2,4,5]
    ϕ = ϕ/sum(ϕ)
    y_samp = sample_cat(ϕ,100_000)
    figure()
    hist(y_samp,(1:5) .- 0.5,normed=true)
    stem(1:4,ϕ,"r")
end

function softmax(λs::Union{Vector{Float64},Array{Float64,2}})
    L = size(λs)
    if length(L) == 1
        exp_λs = exp.(λs)
        return exp_λs./sum(exp_λs)
    else
        ϕs = similar(λs)
        for i in 1:L[1]
            ϕs[i,:] = softmax(λs[i,:])
        end
        return ϕs
    end
end





## create a toy regression example
n = 100
x1 = -2 .+ rand(n).*4
x2 = -2 .+ rand(n).*4


# true parameters
#β1 = [0, 0, 0.0] 
β2 = [3, 5, 1.0]
β3 = [2, 1, 5.0]
β4 = [0, 10, 3.0]

β = hcat(β2,β3,β4).*0.5
const X = hcat(ones(n),x1,x2) 

const λs = hcat(zeros(n), X*β)
const ϕs = softmax(λs)


# generate data
const y = zeros(n)
for i = 1:n
    y[i] = sample_cat(ϕs[i,:])
end


# plot data
colors = [:red,:blue,:orange,:green]
figure()
for i = 1:n
    text(x1[i],x2[i],string(Int(y[i])), color = colors[Int(y[i])])
end
axis([minimum(x1),maximum(x1),minimum(x2),maximum(x2)].*1.3)


struct Likelihood
    y::Union{Vector{Float64},Vector{Int64}}
    X::Array{Float64,2}
    λs::Array{Float64,2}
    ϕs::Array{Float64,2}
    function Likelihood(y,X)
        n = length(y)
        m = 4
        new(y,X, zeros(n,m), zeros(n,m))
    end
     
end



## create the Bayesian sofmax model from scratch

# initial paramteters 
β_init = copy(β) #zeros(size(β))
θ_init = β_init[:]



function log_likelihood(likelihood::Likelihood, θ::Vector{Float64})
    β = reshape(θ,(3,3))
    n = length(y)
    likelihood.λs .= hcat(ones(n),likelihood.X*β)
    likelihood.ϕs .= softmax(likelihood.λs)
    llh = 0.0
    for i in 1:n
        llh += sum((likelihood.y[i] .== 1:4).*log.(likelihood.ϕs[i,:]))
    end
    return llh
end


# log-likelihood categorical
function log_likelihood(θ::Vector{Float64},y::Union{Vector{Float64},Vector{Int64}},X::Array{Float64,2},λs::Array{Float64,2},ϕs::Array{Float64,2})
    β = reshape(θ,(3,3))
    n = length(y)
    λs .= hcat(ones(n),X*β)
    ϕs .= softmax(λs)
    llh = 0.0
    for i in 1:n
        llh += sum(float.(y[i] .== 1:4).*log.(ϕs[i,:]))
    end
    return llh
end
    
# log-prior
function log_prior(β)
    return -0.5*β'*β/10^2
end

# log posterior
likelihood = Likelihood(y,X)
log_posterior(θ) = log_likelihood(likelihood,θ) + log_prior(θ)
# log_posterior(θ) = log_likelihood(θ,y,X,λs,ϕs) + log_prior(θ)


θs = β[:]
## sample the posterior
N_burn_in  = 500
θs_samp, lps = sample(copy(θ_init), ones(length(θ_init)), log_posterior, 10_000, N_burn_in)


# remove "burn in" phase
θs_samp = θs_samp[:,N_burn_in+1:end]
lps = lps[N_burn_in+1:end] 



# plot mcmc chain
figure()
for n=1:9
    subplot(9,2,2*n-1), plot(θs_samp[n,:]), plot(1:length(θs_samp[n,:]),θs[n]*ones(length(θs_samp[n,:])))
    subplot(9,2,2*n), hist(θs_samp[n,:],1000), plot(θs[n],0,"ro")
end
tight_layout()