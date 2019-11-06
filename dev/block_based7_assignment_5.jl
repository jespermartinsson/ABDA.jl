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

logy = 1.0*y
for j in 1:length(y)
    logy[j] = log.(y[j])
end

logy_flatten = vcat(logy...)
mean_logy = mean(logy_flatten)
std_logy = std(logy_flatten)

zlogy = similar(logy)
for j in 1:length(logy)
    zlogy[j] = (logy[j] .- mean_logy)./std_logy
end



figure()
for j in 1:length(y)
    i = 1:length(y[j])
    plot(i, y[j], ".-", alpha = .5)
end



function log_ll(θ::Vector{Float64}, σ::Vector{Float64}, y::Vector{Float64})
    if σ[1] <= 0.0
        return -Inf
    else
        ε = y .- θ[1]
        return -length(ε)*log(σ[1]) - 0.5*ε'ε/σ[1]^2
    end
end


function log_prior_theta(θ::Vector{Float64}, μ::Vector{Float64}, τ::Vector{Float64})
    if τ[1] <= 0.0
        return -Inf
    else
        return sum(-log(τ[1]) .- 0.5 .* ((θ .- μ[1])./τ[1]).^2)
    end
end

function log_prior_sigma(σ::Vector{Float64})
    if σ[1] <= 0.0
        return -Inf 
    else
        return 0.0
    end
end

function log_prior_tau(τ::Vector{Float64})
    if τ[1] <= 0.0
        return -Inf 
    else
        return 0.0
    end
end

function log_prior_mu(μ::Vector{Float64})
    return 0.0
end

mutable struct Posterior
    y::Array{Array{Float64,1},1}
    Posterior() = new()
    Posterior(y::Array{Array{Float64,1},1}) = new(y)
end

function log_pdf(p::Posterior, θs::Vector{Vector{Float64}}, θ::Vector{Float64}, j::Int64)
    J = length(p.y)
    if 1 <= j <= J
        σ = θs[J+1]
        μ = θs[J+2]
        τ = θs[J+3]
        return log_ll(θ, σ, p.y[j]) + log_prior_theta(θ, μ, τ) + log_prior_mu(μ) + log_prior_sigma(σ) + log_prior_tau(τ)
    end
    if j == J+1
        σ = θ
        return log_pdf(p::Posterior, θs) + log_prior_sigma(σ)
    end
    if j == J+2
        μ = θ
        τ = θs[J+3]
        return sum([log_prior_theta(θs[j], μ, τ) for j in 1:length(θs)]) + log_prior_mu(μ)
    end
    if j == J+3
        μ = θs[J+2]
        τ = θ
        if τ[1] <= 0
            return -Inf
        else
            return sum([log_prior_theta(θs[j], μ, τ) for j in 1:length(θs)]) + log_prior_tau(τ)
        end
    end
end


function log_pdf(p::Posterior, θs::Vector{Vector{Float64}})
    J = length(p.y)
    σ = θs[J+1]
    μ = θs[J+2]
    τ = θs[J+3]

    value = 0.0
    for j = 1:J
        value += log_ll(θs[j], σ, p.y[j]) + log_prior_theta(θs[j], μ, τ)
    end
    value += + log_prior_mu(μ) + log_prior_sigma(σ) + log_prior_tau(τ)

    return value
end





J = length(y)
posterior = Posterior(zlogy)

θs = Vector{Vector{Float64}}(undef,J+3)
w = Vector{Vector{Float64}}(undef,J+3)
for j in 1:J+3
    θs[j] = rand(1)
    w[j] = ones(1)
end

log_pdf2(θs) = log_pdf(posterior,θs)
log_pdf2(θs::Vector{Vector{Float64}}, θ::Vector{Float64}, j::Int64) = log_pdf(posterior,θs,θ,j)


using BenchmarkTools

@btime log_pdf2(θs)
j = 3
@btime log_pdf2(θs,θs[j],j)



Random.seed!(1)
@time xs1, lp1 = block_sample(deepcopy(θs), deepcopy(w),  log_pdf2, 10_000; printing=true)
@time xs2, lp2 = block_fsample(deepcopy(θs), deepcopy(w),  log_pdf2, 100_000; printing=true)


b = 2
figure()
plot(xs1[b][1,:],".-")
figure()
plot(xs2[b][1,:],".-")
figure()
for b in 1:length(xs1)
println(ABDA.ess(xs1[b]))
println(ABDA.ess(xs2[b]))
println(ABDA.ess(xs1[b])./ABDA.ess(xs2[b]))
plot(ABDA.ess(xs1[b])./ABDA.ess(xs2[b]),"o")
end