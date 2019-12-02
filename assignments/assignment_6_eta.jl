using ABDA
using PyPlot
using Statistics
using BenchmarkTools

run_mcmc = true


script_name = split(@__FILE__, "/")[end]
pathdir = prod([k * "/" for k in split(@__FILE__, "/")[1:end - 1]])


y = [607.0, 583, 521, 494, 369, 782, 570, 678, 467, 620, 425, 395, 346, 361, 310, 300, 382, 294, 315, 323, 421, 339, 398, 328, 335, 291, 329, 310, 294, 321, 286, 349, 279, 268, 293, 310, 259, 241, 243, 272, 247, 275, 220, 245, 268, 357, 273, 301, 322, 276, 401, 368, 149, 507, 411, 362, 358, 355, 362, 324, 332, 268, 259, 274, 248, 254, 242, 286, 276, 237, 259, 251, 239, 247, 260, 237, 206, 242, 361, 267, 245, 331, 357, 284, 263, 244, 317, 225, 254, 253, 251, 314, 239, 248, 250, 200, 256, 233, 427, 391, 331, 395, 337, 392, 352, 381, 330, 368, 381, 316, 335, 316, 302, 375, 361, 330, 351, 186, 221, 278, 244, 218, 126, 269, 238, 194, 384, 154, 555, 387, 317, 365, 357, 390, 320, 316, 297, 354, 266, 279, 327, 285, 258, 267, 226, 237, 264, 510, 490, 458, 425, 522, 927, 555, 550, 516, 548, 560, 545, 633, 496, 498, 223, 222, 309, 244, 207, 258, 255, 281, 258, 226, 257, 263, 266, 238, 249, 340, 247, 216, 241, 239, 226, 273, 235, 251, 290, 473, 416, 451, 475, 406, 349, 401, 334, 446, 401, 252, 266, 210, 228, 250, 265, 236, 289, 244, 327, 274, 223, 327, 307, 338, 345, 381, 369, 445, 296, 303, 326, 321, 309, 307, 319, 288, 299, 284, 278, 310, 282, 275, 372, 295, 306, 303, 285, 316, 294, 284, 324, 264, 278, 369, 254, 306, 237, 439, 287, 285, 261, 299, 311, 265, 292, 282, 271, 268, 270, 259, 269, 249, 261, 425, 291, 291, 441, 222, 347, 244, 232, 272, 264, 190, 219, 317, 232, 256, 185, 210, 213, 202, 226, 250, 238, 252, 233, 221, 220, 287, 267, 264, 273, 304, 294, 236, 200, 219, 276, 287, 365, 438, 420, 396, 359, 405, 397, 383, 360, 387, 429, 358, 459, 371, 368, 452, 358, 371]

const ind = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34]


logy = log.(y)
const J = 34
const zlogy = (logy .- mean(logy)) ./ std(logy)
const I = length(y)

const child_j = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
const child_i = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    #cd(dirname(PROGRAM_FILE))
    #pathdir = "/media/jesper/ssd/LTU/research/olin/julia/"
using PyPlot
close("all")

function logprior_theta(eta::Array{Float64}, mu::Float64, tau::Float64)
    return sum(-log.(tau) .- 0.5 * ((eta .- mu) ./ tau).^2)
end

function logprior_sigma(sigma::Float64)
    return log(sigma > 0)
end

function logprior_tau(tau::Float64)
    return log(tau > 0)
end

function logprior_mu(mu::Float64)
    return 0
end

function logll(beta::Array{Float64})
    eta = beta[1:J]
    b0 = beta[J + 1]
    b1 = beta[J + 2]
    sigma = beta[J + 3]
    tau = beta[J + 4]

    mu = b0 .+ b1 * child_j
    theta = mu .+ tau * eta

    lp = 0.0
    for i in 1:I
        j = ind[i]
        lp += -log(sigma) - 0.5 * ((zlogy[i] - theta[j]) / sigma)^2
    end
    return lp
end

function logpost(beta::Array{Float64})
    eta = beta[1:J]
    sigma = beta[J + 3]
    tau = beta[J + 4]

    if (sigma > 0) && (tau > 0)
        return logll(beta) + logprior_theta(eta, 0.0, 1.0) + logprior_sigma(sigma) + logprior_tau(tau)
    else
        return -Inf
    end

end



beta = ones(J + 4)
#beta[J+2]=0.1
#@enter logpost(beta)
#error()

if run_mcmc
    @time zeta_samp, lp = sample(beta, ones(length(beta)), logpost, 10_0000; printing=true)
end



# j = 1
# y_j = []
# y_i = []
# for i in 1:I
#     if ind[i] == j
#         push!(y_i, y[i])
#     else
#         push!(y_j, y_i)
#         y_i = []
#         push!(y_i, y[i])
#         j += 1
#     end
# end
# push!(y_j, y_i)

# figure()
# col = ""
# for j in 1:J
#     if child_j[j] == 1
#         col = "r"
#     else
#         col = "k"
#     end
#     subplot(121)
#     plot(1:length(y_j[j]), y_j[j], col * "-")
#     ylabel("reaction time")
#     xlabel("attempt nr")
#     xlim([1,10])
#     subplot(122)
#     plot(1:length(y_j[j]), log.(y_j[j]), col * "-")
#     ylabel("log(reaction time)")
#     xlabel("attempt nr")
#     xlim([1,10])
# end
# tight_layout()





eta_samp = zeta_samp[1:J,:]
b0_samp = zeta_samp[J + 1,:]
b1_samp = zeta_samp[J + 2,:]
sigma_samp = zeta_samp[J + 3,:]
tau_samp = zeta_samp[J + 4,:]
mu_samp = repeat(b0_samp, 1, J)' .+ repeat(b1_samp, 1,J)' .* repeat(child_j, 1, size(b1_samp, 1))
theta_samp = mu_samp .+ eta_samp .* repeat(tau_samp, 1, J)'


figure()
subplot(121)
ABDA.hist(b1_samp)
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


subplot(122)
ABDA.hist(std(logy) * b1_samp)
xlabel(raw"$\varphi$")
tight_layout()





# (ln(y)-my)/sy = theta + s*e
# ln(y) = sy*(theta + s*e) + my
# ln(y) = beta + sigma*e, beta = sy*theta+my, sigma = sy*s
# y = exp(sy*theta + my + sy*s*e)
# y = exp(sy*theta+my)*exp(sy*s*e)
# y = exp(theta2)*exp(sigma2*e)

sigma2_samp = std(logy) * sigma_samp
theta2_samp = std(logy) * theta_samp + mean(logy)

mu_y_samp = exp.(theta2_samp + repeat(sigma2_samp', size(theta2_samp, 1), 1).^2 / 2)


figure()
N = Int(ceil(sqrt(J)))
for i in 1:J
    subplot(N, N, i)
    ABDA.hist(mu_y_samp[i,:])
    xlabel(string("\$\\theta_{", i, "}\$"))
end
tight_layout()

mu2_samp = std(logy) * mu_samp + mean(logy)
tau2_samp = std(logy) * tau_samp

exp_mu2_samp = exp.(mu2_samp + repmat(tau2_samp, 1, J)'.^2 / 2 + repmat(sigma2_samp, 1, J)'.^2 / 2)

figure()
N = Int(ceil(sqrt(J)))
i = 0
for i in 1:J
    subplot(N, N, i)
    ABDA.hist(exp_mu2_samp[i,:])
    xlabel(string("\$\\theta_{", i, "}\$"))
end
tight_layout()


#eta = beta[1:J]
#b0 = beta[J+1]
#b1 = beta[J+2]
#sigma = beta[J+3]
#tau = beta[J+4]
#mu = b0 + b1*child_j
#theta = mu + tau*eta

figure()
subplot(121)
ABDA.hist(tau_samp)
xlabel(string("\$\\tilde{\\tau}\$"))
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
ABDA.hist(std(logy) * tau_samp)
xlabel(string("\$\\tau\$"))
tight_layout()


using Distributions
a, b = 1, 1
if true
    z = sum(child_j)
else
    z = Int(round(J / 2)) # test half kids half adults
end
thetas = rand(Beta(z + a, J - z + b), size(zeta_samp, 2))
child_rand = thetas[i] .> rand(size(zeta_samp, 2))



my = mean(logy)
sy = std(logy)

y_pred = zeros(size(zeta_samp, 2), 3)
for i in 1:size(zeta_samp, 2)
    child_i = 0.0
    mu_i = b0_samp[i] + b1_samp[i] * child_i
    theta_i = mu_i + tau_samp[i] * randn()
    logzy_i = theta_i + sigma_samp[i] * randn()
    logy_i = sy * logzy_i + my
    y_pred[i,1] = exp(logy_i)

    child_i = 1.0
    mu_i = b0_samp[i] + b1_samp[i] * child_i
    theta_i = mu_i + tau_samp[i] * randn()
    logzy_i = theta_i + sigma_samp[i] * randn()
    logy_i = sy * logzy_i + my
    y_pred[i,2] = exp(logy_i)

    child_i = child_rand[i]
    mu_i = b0_samp[i] + b1_samp[i] * child_i
    theta_i = mu_i + tau_samp[i] * randn()
    logzy_i = theta_i + sigma_samp[i] * randn()
    logy_i = sy * logzy_i + my
    y_pred[i,3] = exp(logy_i)
end

figure()
ABDA.hist(y_pred[:,1], color = "b")
ABDA.hist(y_pred[:,2], color = "r")
ABDA.hist(y_pred[:,3], color = "k")
title("posterior predicted reaction time a random individual")






mle = []
for j in 1:J
    push!(mle, mean(logy[ind .== j]))
end

function logprior_thetas(theta, mu, tau) # p(theta[1],...,theta[J]) ~ N(mu,tau)
    return (-log(tau) - 0.5 * ((theta - mu) / tau).^2)
end


figure()
for j in 1:J
#    fr,bins = np.histogram(theta2_samp[j,:],Int(round(sqrt(length(theta2_samp[j,:])))),normed=true)
#    x,y = get_mystep(bins,fr)
#    plot(x,y*0.3+j)
    if child_j[j] == 0
        ABDA.hist(theta2_samp[j,:], baseline = j * 10, color = "b")
    else
        ABDA.hist(theta2_samp[j,:], baseline = j * 10, color = "r")
    end
    plot(mle[j], j * 10, "ro")
end
plot(mle[1], 1 * 10, "ro", label = "MLE estimate \$\\hat{\\theta}_{\\mathrm{MLE}}\$")
xl = plt["xlim"]()
thetas = linspace(xl[1], xl[2], 1000)
plot(thetas, 80 * exp.(logprior_thetas(thetas, mean(mu2_samp, 2)[1], mean(tau2_samp))), label = "\$p(\\hat{\\mu},\\hat{\\tau)}\$", color = "k")
plot(thetas, 80 * exp.(logprior_thetas(thetas, mean(mu2_samp, 2)[2], mean(tau2_samp))), label = "\$p(\\hat{\\mu}+\\hat{\\varphi},\\hat{\\tau)}\$", color = "r")
plt["legend"]()
plt["yticks"]([])
xlabel(string("\$\\theta_j\$ expected log(reaction time)"))
plt["tight_layout"]()


figure()
subplot(121)
ABDA.hist(sigma_samp)
xlabel(string("\$\\tilde{\\sigma}\$"))
plt["tight_layout"]()
subplot(122)
ABDA.hist(std(logy) * sigma_samp)
xlabel(string("\$\\sigma\$"))
plt["tight_layout"]()