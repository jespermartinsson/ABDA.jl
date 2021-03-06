using ABDA
using PyPlot
using Random
using Statistics
using LinearAlgebra
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
    n = 10
    X = hcat(ones(n), collect(1:n), collect(1:n).^2)
    β = [5,5,1] .+ [1,1,0.1].*randn(3)
    σ = 0.5 + 1*rand()
    y = X*β .+ σ*randn(n)

    push!(ys,y)
    push!(Xs,X)
    push!(βs,β)
    push!(σs,σ)
    push!(θs,vcat(β,σ))
end

# with standardized data
if false
    ys_flatten = vcat(ys...)
    mean_ys = mean(ys_flatten)
    std_ys = std(ys_flatten)

    zys = deepcopy(ys)
    for j in 1:length(ys)
        zys[j] = (ys[j] .- mean_ys)./std_ys
    end
    ys .= zys

    Xs_flatten = vcat(Xs...)
    mean_Xs = mean(Xs_flatten,dims=1)
    std_Xs = std(Xs_flatten,dims=1)
    
    zXs = deepcopy(Xs)
    for j in 1:length(Xs)
        for k = 2:length(mean_Xs)
            zXs[j][:,k] = (Xs[j][:,k] .- mean_Xs[k])./std_Xs[k]
        end
    end
    Xs .= zXs
end

figure()
for j in 1:length(ys)
    i = 1:length(ys[j])
    plot(Xs[j][:,2], ys[j], ".-", alpha = .5)
end

function log_ll(θ::Vector{Float64}, y::Vector{Float64}, X::Array{Float64,2})
    if θ[end] <= 0.0
        return -Inf
    else
        ε = y .- X*θ[1:end-1]
        return -length(ε)*log(θ[end]) - 0.5*ε'ε/θ[end]^2
    end
end


function log_prior_theta(θ::Vector{Float64}, μ::Vector{Float64}, τ::Vector{Float64})
    if any(τ .<= 0.001) || θ[end] <= 0
        return -Inf
    else
        value = 0.0
        for k in 1:length(θ)
            value += -log(τ[k]) .- 0.5 .* ((θ[k] .- μ[k])./τ[k]).^2
        end
        return value
    end
end

function log_prior_sigma(σ::Vector{Float64})
    if any(σ .<= 0.0)
        return -Inf 
    else
        return 0.0
    end
end

function log_prior_tau(τ::Vector{Float64})
    if any(τ .<= 0.0)
        return -Inf 
    else
        return 0.0
    end
end

function log_prior_mu(μ::Vector{Float64})
    return 0.0
end

mutable struct Posterior
    ys::Array{Array{Float64,1},1}
    Xs::Array{Array{Float64,2},1}
    Posterior() = new()
    Posterior(ys::Array{Array{Float64,1},1},Xs::Array{Array{Float64,2},1}) = new(ys,Xs)
end

function log_pdf(p::Posterior, θs::Vector{Vector{Float64}}, j::Int64)
    J = length(p.ys)
    if 1 <= j <= J
        μ = θs[J+1]
        τ = θs[J+2]
        return (θ::Vector{Float64}) -> log_ll(θ, p.ys[j], p.Xs[j]) + log_prior_theta(θ, μ, τ) + log_prior_mu(μ) + log_prior_tau(τ)
    end
    if j == J+1
        τ = θs[J+2]
        return (μ::Vector{Float64}) -> sum([log_prior_theta(θs[k], μ, τ) for k in 1:J]) + log_prior_mu(μ)
    end
    if j == J+2
        μ = θs[J+1]
        return (τ::Vector{Float64}) -> τ[1] > 0.0 ? sum([log_prior_theta(θs[k], μ, τ) for k in 1:J]) + log_prior_tau(τ) : -Inf
    end
end

function log_pdf(p::Posterior, θs::Vector{Vector{Float64}})
    J = length(p.ys)
    μ = θs[J+1]
    τ = θs[J+2]

    value = 0.0
    for j = 1:J
        value += log_ll(θs[j], p.ys[j], p.Xs[j]) + log_prior_theta(θs[j], μ, τ)
    end
    value += log_prior_mu(μ) + log_prior_tau(τ)

    return value
end





J = length(ys)
posterior = Posterior(ys,Xs)

θs = Vector{Vector{Float64}}(undef,J+2)
w = Vector{Vector{Float64}}(undef,J+2)
for j in 1:J
    θs[j] = rand(4)
    w[j] = ones(4)
end
j = J+1
θs[j] = ones(4)
w[j] = ones(4)
j = J+2
θs[j] = ones(4)
w[j] = ones(4)


log_pdf2(θs) = log_pdf(posterior,θs)
log_pdf2(θs::Vector{Vector{Float64}}, j::Int64) = log_pdf(posterior,θs,j)


function b2v(θs)
    return vcat(θs...)
end

function v2b(θ)
    J = 30
    θs = Vector{Vector{Float64}}(undef,J+3)
    for j in 1:J
        θs[j] = θ[j*2-1:j*2]
    end
    j = J+1
    θs[j] = θ[2*J+1:2*J+2]
    j = J+2
    θs[j] = θ[2*J+3:2*J+4]
    return θs
end


log_pdf3(θ) = log_pdf(posterior,v2b(θ))


using BenchmarkTools

@btime log_pdf2(θs)
j = 3
@btime log_pdf2(θs,j)(θs[j])
for j in J+1:J+5
@btime log_pdf2(θs,j)(θs[j])
end


Random.seed!(2)
#@time xs1, lp1 = sample(deepcopy(b2v(θs)), deepcopy(b2v(w)),  log_pdf3, 10_000, 1_000; printing=true)
@time xs1, lp1 = ABDA.block_sample(deepcopy(θs), deepcopy(w),  log_pdf2, 10_000, 1_000; printing=true)
@time xs2, lp2 = ABDA.block_fsample(deepcopy(θs), deepcopy(w),  log_pdf2, 10_000, 1_000; printing=true)
#@time xs1, lp1 = ABDA.block_slice_sample(deepcopy(θs), deepcopy(w),  log_pdf2, 10_000; printing=true)
#@time xs2, lp2 = ABDA.block_rslice_sample(deepcopy(θs), log_pdf2, 10_000; ws0 = deepcopy(w), printing=true)




b = 3
figure(), plot3D(xs2[b][1,:],xs2[b][2,:],xs2[b][3,:],".")

b = 1
for k in 1:4
figure()
subplot(221), plot(xs1[b][k,:],"k-")
subplot(222), plot(xs2[b][k,:],"r-")
subplot(223), plot(xs1[b][k,100:300],"k-")
subplot(224), plot(xs2[b][k,100:300],"r-")
end

figure()
for b in 1:length(xs1)
println(ABDA.ess(xs1[b]))
println(ABDA.ess(xs2[b]))
println(ABDA.ess(xs1[b])./ABDA.ess(xs2[b]))
plot(ABDA.ess(xs1[b])./ABDA.ess(xs2[b]),"o")
end


for b in J+1:length(xs1)
figure()
plot(xs1[b]',"g-",alpha=0.5)
plot(xs2[b]',"r-",alpha=0.5)
end