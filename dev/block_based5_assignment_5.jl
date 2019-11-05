using ABDA
using PyPlot
using Random
using Statistics
using LinearAlgebra




close("all")

include("parser_5.jl")

# read data from csv
filename = (@__DIR__) * raw"/../assignments/data/ABDA 2019 -- Reaction time - Sheet1.tsv"
skip_lines = [14,15,24,25,26,27,42] .- 1 
y, subject = parse_data_array(filename, skip_lines)



figure()
for j in 1:length(y)
    i = 1:length(y[j])
    plot(i, y[j], ".-", alpha = .5)
end



mutable struct Likelihood # individual likelihood
    y::Vector{Int64}
    # constructors
    Likelihood() = new()
    Likelihood(y) = new(y)
end
function log_pdf(l::Likelihood, θ::Vector{Float64})
    if any(θ .<= 0)
        return -Inf
    else
        ε = l.y .- θ[1]
        return -length(ε)*log(θ[end]) - 0.5*ε'ε/θ[end]^2
    end
end

mutable struct Likelihoods # all likelihoods
    lhs::Vector{Likelihood}
    # constructors
    Likelihoods() = new()
    function Likelihoods(y)
        J = length(y)
        lhs = Vector{Likelihood}(undef,J)
        for j in 1:J
            lhs[j] = Likelihood(y[j])
        end
        new(lhs)
    end
end
function log_pdf(l::Likelihoods, θs::Vector{Vector{Float64}})
    value = 0.0
    for j = 1:length(l.lhs)
        value += log_pdf(l.lhs[j], θs[j])
    end
    return value
end
function log_pdf(l::Likelihoods, θ::Vector{Float64},j::Int64)
    return log_pdf(l.lhs[j], θ) 
end




mutable struct Prior # individual prior
    # constructors
    Prior() = new()
end
function log_pdf(p::Prior, θ::Vector{Float64}, ϕ::Vector{Float64})
    if any(θ .<= 0)
        return -Inf
    else
        ε = θ[1:2] .- ϕ[1]
        return -length(ε)*log(ϕ[end]) - 0.5*ε'ε/ϕ[end]^2 
    end
end



mutable struct Priors
    priors::Vector{Prior}
    Priors() = new()
    Priors(priors::Vector{Prior}) = new(priors::Vector{Prior})
end
function log_pdf(p::Priors, θ::Vector{Float64},j::Int64)
    return log_pdf(p.priors[j], θ, [0.0,1000.0]) 
end



mutable struct Posterior
    lhs::Likelihoods
    priors::Priors
    Posterior() = new()
    Posterior(lhs::Likelihoods, priors::Priors) = new(lhs, priors)
end
function log_pdf(p::Posterior, θ::Vector{Float64}, j::Int64)
    return log_pdf(p.lhs, θ, j) + log_pdf(p.priors, θ, j) 
end
function log_pdf(p::Posterior, θs::Vector{Vector{Float64}})
    value = 0.0
    for j = 1:length(p.lhs.lhs)
        value += log_pdf(p, θs[j], j)
    end
    return value
end





J = length(y)
lhs = Likelihoods(y)
priors = Priors([Prior() for j = 1:J])
posterior = Posterior(lhs,priors)

θs = Vector{Vector{Float64}}(undef,J)
w = Vector{Vector{Float64}}(undef,J)
for j in 1:J
    θs[j] = rand(2)
    w[j] = ones(2)
end

log_pdf2(θs) = log_pdf(posterior,θs)
log_pdf2(θ::Vector{Float64}, j::Int64) = log_pdf(posterior,θ,j)

log_pdf3(θs) = log_pdf(lhs,θs)
log_pdf3(θ::Vector{Float64}, j::Int64) = log_pdf(lhs,θ,j)

Random.seed!(1)
@time xs1, lp1 = block_sample(deepcopy(θs), deepcopy(w),  log_pdf2, 10_000; printing=true)
@time xs2, lp2 = block_fsample(deepcopy(θs), deepcopy(w),  log_pdf2, 10_000; printing=true)


b = 2
figure()
plot(xs1[b][1,:],".-")
plot(xs1[b][2,:],".-")
figure()
plot(xs2[b][1,:],".-")
plot(xs2[b][2,:],".-")
figure()
for b in 1:length(xs1)
println(ABDA.ess(xs1[b]))
println(ABDA.ess(xs2[b]))
println(ABDA.ess(xs1[b])./ABDA.ess(xs2[b]))
plot(ABDA.ess(xs1[b])./ABDA.ess(xs2[b]),"o")
end