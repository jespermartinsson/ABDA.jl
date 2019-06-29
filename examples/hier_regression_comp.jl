using ABDA
using Statistics
using PyPlot

using BenchmarkTools

close("all")

## create a toy regression example

J = 300
βs = Vector{Vector{Float64}}(undef,0)
σs = Vector{Float64}(undef,0)
θs = Vector{Vector{Float64}}(undef,0)
Xs = Vector{Array{Float64,2}}(undef,0)
ys = Vector{Vector{Float64}}(undef,0)
for j in 1:J
    n = 5
    X = hcat(ones(n),1:n)
    β = [1,2] .+ randn(2)
    σ = 1
    y = X*β .+ σ*randn(n)

    push!(ys,y)
    push!(Xs,X)
    push!(βs,β)
    push!(σs,σ)
    push!(θs,vcat(β,σ))
end

figure()
plot(Xs[1][:,2],ys[1],"o-")



function iseq(a::Vector{Float64},b::Vector{Float64})
    for n in 1:length(a)
        if a[n] != b[n]
            return false
        end
    end
    return true
end


mutable struct LogLikelihood
    y::Vector{Float64}
    X::Array{Float64,2}
    θ::Vector{Float64}
    value::Float64
    function LogLikelihood(y,X)
        new(y,X,NaN*zeros(3),NaN)
    end
end
function (llh::LogLikelihood)(θ::Vector{Float64})
    if !iseq(θ,llh.θ)
        llh.θ .= θ 
        ε = (llh.y .- llh.X*llh.θ[1:end-1])
        llh.value = -length(ε)*log(θ[end]) - 0.5*ε'*ε/llh.θ[end]^2
    end
    return llh.value
end


mutable struct LogLikelihoods
    vllh::Vector{LogLikelihood}
    value::Float64
    function LogLikelihoods(vllh)
        new(vllh,NaN)
    end
end
function (llhs::LogLikelihoods)(θs::Vector{Vector{Float64}})
    llhs.value = 0.0
    for j = 1:length(llhs.vllh)
        llhs.value += llhs.vllh[j](θs[j])
    end
    return llhs.value
end


using LinearAlgebra

mutable struct LogPrior
    μ::Vector{Float64}
    C::Array{Float64,2}
    iC::Array{Float64,2}
    dC::Float64
    θ::Vector{Float64}
    value::Float64
    function LogPrior(μ,C)
        new(μ,C,inv(C),det(C),NaN*zeros(2),NaN)
    end
end
function (lp::LogPrior)(θ::Vector{Float64})
    if θ==lp.θ
        return lp.value
    else
        lp.θ .= θ 
        ε = lp.θ .- θ
        lp.value = -0.5*log(lp.dC) - 0.5*ε'*lp.iC*ε
        return lp.value
    end
end


mutable struct LogPriors
    vlprior::Vector{LogPrior}
end
function (p::LogPriors)(θs::Vector{Vector{Float64}})
    return 0.0
end




mutable struct LogPosterior
    llhs::LogLikelihoods
    lpriors::LogPriors
    value::Float64
end
function (p::LogPosterior)(θs::Vector{Vector{Float64}})
    return p.llhs(θs) + p.lpriors(θs)
end
function (p::LogPosterior)(θ::Vector{Float64},j)
    return p.llhs(θ,j) + p.lpriors(θ,j)
end






vllh = Vector{LogLikelihood}(undef,0)
for j = 1:J
    llh = LogLikelihood(ys[j],Xs[j])
    push!(vllh,llh)
end
llhs = LogLikelihoods(vllh)


@btime llhs(θs)
θs[1] .+= 0.01
@btime llhs(θs)
θs = [θ .+ 0.01 for θ in θs]
@btime llhs(θs)



error()



# true parameters

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