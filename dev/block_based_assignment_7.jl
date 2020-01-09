using ABDA
using PyPlot
using Random
using Statistics
using LinearAlgebra




close("all")

include("parser_7.jl")

# read data from csv
filename = (@__DIR__) * raw"/../assignments/data/ABDA 2019 -- Reaction time - Sheet1.tsv"
skip_lines = [14,15,24,25,26,27,42] .- 1 
y, subject, ischild, x = parse_data_array(filename, skip_lines)

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

x_flatten = vcat(x...)
mean_x = mean(x_flatten)
std_x = std(x_flatten)

zx = Vector{Vector{Float64}}(undef,length(x))
for j in 1:length(x)
    zx[j] = (x[j] .- mean_x)./std_x
end

figure()
for j in 1:length(y)
    i = 1:length(y[j])
    plot(i, y[j], ".-", alpha = .5)
end



function log_ll(θ::Vector{Float64}, σ::Vector{Float64}, y::Vector{Float64}, x::Vector{Float64})
    if σ[1] <= 0.0
        return -Inf
    else
        ε = y .- (θ[1] .+ θ[2].*x)
        return -length(ε)*log(σ[1]) - 0.5*ε'ε/σ[1]^2
    end
end


function log_prior_theta(θ::Vector{Float64}, μ::Vector{Float64}, τ::Vector{Float64}, φ::Vector{Float64}, ischild::Int64)
    if any(τ .<= 0.0)
        return -Inf
    else
        μ_θ = μ .+ φ*ischild
        ε = (θ .- μ_θ)./τ
        return sum(-log.(τ) .- 0.5 .* ε.^2)
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
    if any(τ .<= 0.0)
        return -Inf 
    else
        return 0.0
    end
end

function log_prior_mu(μ::Vector{Float64})
    return 0.0
end

function log_prior_phi(φ::Vector{Float64})
    return 0.0
end

struct Posterior
    y::Array{Array{Float64,1},1}
    ischild::Array{Int64,1}
    x::Array{Array{Float64,1},1}
    Posterior(y::Array{Array{Float64,1},1},ischild::Array{Int64,1},x::Array{Array{Float64,1},1}) = new(y,ischild,x)
end


function log_pdf(p::Posterior, θs::Vector{Vector{Float64}}, j::Int64)::Function
    J = length(p.y)
    if 1 <= j <= J
        σ = θs[J+1]
        μ = θs[J+2]
        τ = θs[J+3]
        φ = θs[J+4]
        return (θ::Vector{Float64}) -> log_ll(θ, σ, p.y[j], p.y[j]) + log_prior_theta(θ, μ, τ, φ, p.ischild[j]) + log_prior_mu(μ) + log_prior_sigma(σ) + log_prior_tau(τ) + log_prior_phi(φ)
    end
    if j == J+1
        return (σ::Vector{Float64}) -> sum([log_ll(θs[k], σ, p.y[k], p.x[k]) for k in 1:J]) + log_prior_sigma(σ)
    end
    if j == J+2
        τ = θs[J+3]
        φ = θs[J+4]
        return (μ::Vector{Float64}) -> sum([log_prior_theta(θs[k], μ, τ, φ, p.ischild[k]) for k in 1:J]) + log_prior_mu(μ)
    end
    if j == J+3
        μ = θs[J+2]
        φ = θs[J+4]
        return (τ::Vector{Float64}) -> any(τ .> 0.0) ? sum([log_prior_theta(θs[k], μ, τ, φ, p.ischild[k]) for k in 1:J]) + log_prior_tau(τ) : -Inf
    end
    if j == J+4
        μ = θs[J+2]
        τ = θs[J+3]
        return (φ::Vector{Float64}) -> sum([log_prior_theta(θs[k], μ, τ, φ, p.ischild[k]) for k in 1:J]) + log_prior_phi(φ)
    end
end


function log_pdf(p::Posterior, θs::Vector{Vector{Float64}})
    J = length(p.y)
    σ = θs[J+1]
    μ = θs[J+2]
    τ = θs[J+3]
    φ = θs[J+4]

    value = 0.0
    for j = 1:J
        value += log_ll(θs[j], σ, p.y[j],p.x[j]) + log_prior_theta(θs[j], μ, τ, φ, p.ischild[j])
    end
    value += + log_prior_mu(μ) + log_prior_sigma(σ) + log_prior_tau(τ) + log_prior_phi(φ)

    return value
end


J = length(y)
posterior = Posterior(zlogy, ischild, zx)

θs = Vector{Vector{Float64}}(undef,J+4)
w = Vector{Vector{Float64}}(undef,J+4)
for j in 1:J
    θs[j] = rand(2)
    w[j] = ones(2)
end
j = J+1
θs[j] = rand(1)
w[j] = ones(1)

j = J+2
θs[j] = rand(2)
w[j] = ones(2)

j = J+3
θs[j] = rand(2)
w[j] = ones(2)

j = J+4
θs[j] = rand(2)
w[j] = ones(2)

log_pdf2(θs) = log_pdf(posterior,θs)
log_pdf2(θs::Vector{Vector{Float64}}, θ::Vector{Float64}, j::Int64) = log_pdf(posterior,θs,θ,j)
log_pdf2(θs::Vector{Vector{Float64}}, j::Int64) = log_pdf(posterior,θs,j)



Random.seed!(1)
#@time xs1, lp1 = block_sample(deepcopy(θs), deepcopy(w),  log_pdf2, 10_000; printing=true)
N = 100_000
@time zeta_samp_block, lp = ABDA.block_fsample(deepcopy(θs), deepcopy(w),  log_pdf2, N; printing=true)

zeta_samp_block

theta_samp = zeros(J,N)
for j in 1:J
    theta_samp[j,:] .= zeta_samp_block[j][1,:]
end

sigma_samp = zeta_samp_block[J + 1][1,:]
mu_samp = zeta_samp_block[J + 2][1,:]
tau_samp = zeta_samp_block[J + 3][1,:]
phi_samp = zeta_samp_block[J + 4][1,:]

# (ln(y)-my)/sy = theta + s*e
# ln(y) = sy*(theta + s*e) + my
# ln(y) = beta + sigma*e, beta = sy*theta+my, sigma = sy*s
# y = exp(sy*theta + my + sy*s*e)
# y = exp(sy*theta+my)*exp(sy*s*e)
# y = exp(theta2)*exp(sigma2*e)

sigma2_samp = std_logy .* sigma_samp
theta2_samp = std_logy .* theta_samp .+ mean_logy

# mean value of log-normal pdf
# https://en.wikipedia.org/wiki/Log-normal_distribution
mu_y_samp = exp.(theta2_samp .+ repeat(sigma2_samp', size(theta2_samp, 1), 1).^2 ./ 2)



figure(figsize = (8 * 2, 6 * 2), dpi = 80)
B = Int(ceil(sqrt(J)))
col = "b"
for i in 1:J
    subplot(B, B, i)
    x = mu_y_samp[i,:]
    ABDA.hist(x, color = col)
    xlabel(string("\$\\exp(\\theta_{", i, "}+\\sigma^2/2)\$"))
end
tight_layout()

figure()
col = "b"
i = 4
x = mu_y_samp[i,:]
ABDA.hist(x, color = col)
xlabel(string("\$\\exp(\\theta_{", i, "}+\\sigma^2/2)\$"))
title("The expected reaction time for the dude")
tight_layout()





figure()
subplot(121)
ABDA.hist(phi_samp)
xlabel(raw"$\tilde{\varphi}$")
tight_layout()

# (ln(y)-my)/sy = theta + phi*c + s*e
# ln(y) = sy*(theta + phi*c + s*e) + my
# ln(y) = beta + sigma*e, beta = sy*theta + sy*phi*c + my, sigma = sy*s
# y = exp(sy*theta + sy*phi*c + my + sy*s*e)
# y = exp(sy*theta + sy*phi*c +my)*exp(sy*s*e)
# y = exp(theta2+phi2*c)*exp(sigma2*e)
# theta2 = sy*theta + my
# phi2 = sy*phi
phi2_samp = std_logy * phi_samp
subplot(122)
ABDA.hist(phi2_samp)
xlabel(raw"$\varphi$")
tight_layout()

figure()
ABDA.hist(exp.(phi2_samp), label=(raw"$\mathrm{Pr}\{\mathrm{exp}(\varphi)>1\} = $"*string(pval)))
pval = sum(exp.(phi2_samp) .> 1)/N
xlabel(raw"$\mathrm{exp}(\varphi)$")
title("kid's mutiplicative effect on average reaction time")
legend()
tight_layout()





mu2_samp = std_logy .* mu_samp .+ mean_logy
tau2_samp = std_logy .* tau_samp


# Derivation for the expected reaction time for the group
# For a analytical approach let:
# theta = mu + tau*xi
# x = theta + sigma*eta
# where eta~N(0,1) and xi~N(0,1) assuming independence. Then
# x = mu + sigma*eta + tau*xi ~ N(mu, sqrt(sigma^2 + tau^2))
# and sqrt(sigma^2 + tau^2) is the std.
# If y = exp(x) then E(y) = exp(mu + (sigma^2 + tau^2)/2), see also simulation here:
# https://github.com/jespermartinsson/ABDA.jl/blob/master/dev/test_var.jl

exp_mu2_samp = exp.(mu2_samp .+ 0.5 .* tau2_samp.^2 .+ 0.5 .* sigma2_samp.^2)



figure()
ABDA.hist(exp_mu2_samp, color = "b")
xlabel(string("\$\\exp(\\mu + \\tau^2/2 + \\sigma^2/2)\$"))
title("expected reaction time for the group")
tight_layout()




using Distributions
a, b = 1, 1
if true
    z = sum(ischild)
else
    z = Int(round(J / 2)) # test half kids half adults
end
thetas = rand(Beta(z + a, J - z + b), N)
child_rand = thetas[i] .> rand(N)




my = mean_logy
sy = std_logy
y_pred = zeros(N,3)
for i in 1:N
    k = 1
    for child_i in [0,1,child_rand[i]]
        theta_i = (mu_samp[i] + phi_samp[i]*child_i) + tau_samp[i] * randn()
        logzy_i = theta_i + sigma_samp[i] * randn()
        logy_i = sy * logzy_i + my
        y_pred[i,k] = exp(logy_i)
        k += 1
    end
end

figure()
ABDA.hist(y_pred[:,1], color = "k", label="adults")
ABDA.hist(y_pred[:,2], color = "r", label="kids")
ABDA.hist(y_pred[:,3], color = "b", label="mixed")
title("posterior predicted reaction time a random individual")
legend()


mle = []
for j in 1:J
    push!(mle, mean(logy[j]))
end

function logprior_thetas(theta, mu, tau, phi, child) # p(theta[1],...,theta[J]) ~ N(mu,tau)
    return (-log.(tau) .- 0.5 .* ((theta .- (mu .+ phi*child)) ./ tau).^2)
end

offset = 2000
figure()
for j in 1:J
    if ischild[j] == 1
        color = "r"
    else
        color = "k"
    end
    ABDA.hist(theta2_samp[j,:], baseline = j * offset, color = color)
    plot(mle[j], j * offset, "o",color = color)
end
plot(mle[1], 1 * offset, "ko", label = "MLE estimate \$\\hat{\\theta}_{\\mathrm{MLE}}\$")
xl = xlim()
thetas = range(xl[1], stop=xl[2], length=1000)
plot(thetas, 10*offset * exp.(logprior_thetas(thetas, mean(mu2_samp), mean(tau2_samp), mean(phi2_samp),0)), label = "\$p(\\hat{\\mu},\\hat{\\tau)}\$", color = "k")
plot(thetas, 10*offset * exp.(logprior_thetas(thetas, mean(mu2_samp), mean(tau2_samp), mean(phi2_samp),1)), label = "\$p(\\hat{\\mu} + \\hat{\\phi},\\hat{\\tau)}\$", color = "r")
legend()
yticks([])
xlabel(string("\$\\theta_j\$ expected log(reaction time)"))
tight_layout()



figure()
subplot(121)
ABDA.hist(tau_samp)
xlabel(raw"$\tilde{\tau}$")
tight_layout()

# (ln(y)-my)/sy = theta + phi*c + s*e
# ln(y) = sy*(theta + phi*c + s*e) + my
# ln(y) = beta + sigma*e, beta = sy*theta + sy*phi*c + my, sigma = sy*s
# y = exp(sy*theta + sy*phi*c + my + sy*s*e)
# y = exp(sy*theta + sy*phi*c +my)*exp(sy*s*e)
# y = exp(theta2+phi2*c)*exp(sigma2*e)
# theta2 = sy*theta + my
# phi2 = sy*phi


subplot(122)
ABDA.hist(tau2_samp)
xlabel(raw"$\tau$")
tight_layout()



if false
    my = mean(logy)
    sy = std(logy)
    mean_reaction_time = zeros(size(zeta_samp, 2))
    for i in 1:size(zeta_samp, 2)
        theta_i = mu_samp[i] + tau_samp[i] * randn()
        theta2_i = sy * theta_i + my
        mean_reaction_time[i] = exp(theta2_i + 0.5 * sigma2_samp[i]^2)
    end

    figure()
    ABDA.hist(mean_reaction_time)
    title("posterior predicted reaction time for the group")

    figure()
    plot(tau2_samp, sigma2_samp, ".")
end


figure()
subplot(121)
ABDA.hist(sigma_samp)
xlabel(string("\$\\tilde{\\sigma}\$"))
tight_layout()
subplot(122)
ABDA.hist(std_logy * sigma_samp)
xlabel(string("\$\\sigma\$"))
tight_layout()






## Fake data
a, b = 1, 1
if false
    z = sum(ischild)
else
    z = Int(round(J*0.5)) # test half kids half adults
end
thetas = rand(Beta(z + a, J - z + b), N)
child_rand = thetas[i] .> rand(N)

y_pred_fake = zeros(N,3)
for i in 1:N
    k = 1
    for child_i in [0,1,child_rand[i]]
        theta_i = (mu_samp[i] + phi_samp[i]*child_i) + tau_samp[i] * randn()
        logzy_i = theta_i + sigma_samp[i] * randn()
        logy_i = sy * logzy_i + my
        y_pred_fake[i,k] = exp(logy_i)
        k += 1
    end
end

figure()
ABDA.hist(y_pred_fake[:,1], color = "k", label="adults")
ABDA.hist(y_pred_fake[:,2], color = "r", label="kids")
ABDA.hist(y_pred_fake[:,3], color = "b", label="mixed")
title("posterior predicted reaction time a random individual (50 % chidren)")
legend()