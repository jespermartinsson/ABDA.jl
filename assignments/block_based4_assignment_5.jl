using ABDA
using PyPlot
using Random
using Statistics
using LinearAlgebra




close("all")

include("parser_5.jl")

# read data from csv
filename = (@__DIR__) * raw"/data/ABDA 2019 -- Reaction time - Sheet1.tsv"
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


J = length(y)
lhs = Likelihoods(y)
log_pdf(lhs.lhs[1],rand(2))
θs = Vector{Vector{Float64}}(undef,J)
w = Vector{Vector{Float64}}(undef,J)
for j in 1:J
    θs[j] = rand(2)
    w[j] = ones(2)
end
log_pdf(lhs,θs)

log_pdf2(θs) = log_pdf(lhs,θs)
log_pdf2(θ::Vector{Float64}, j::Int64) = log_pdf(lhs,θ,j)
Random.seed!(1)

xs, lp = block_sample(θs, w,  log_pdf2, 10_000; printing=true)


b = 1
figure()
plot(xs[b][1,:],".-")
plot(xs[b][2,:],".-")
