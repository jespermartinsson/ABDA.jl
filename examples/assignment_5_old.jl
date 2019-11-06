using ABDA
using PyPlot
using Statistics
using Random

run_mcmc = true


script_name = split(@__FILE__, "/")[end]
pathdir = prod([k * "/" for k in split(@__FILE__, "/")[1:end - 1]])


y = [ 344.,  307.,  284.,  305.,  297.,  607.,  583.,  521.,  494.,
        369.,  782.,  570.,  678.,  467.,  620.,  517.,  359.,  332.,
        386.,  357.,  286.,  242.,  229.,  298.,  231.,  279.,  317.,
        324.,  303.,  343.,  271.,  266.,  275.,  282.,  281.,  323.,
        380.,  375.,  351.,  304.,  341.,  354.,  275.,  356.,  359.,
        288.,  296.,  292.,  326.,  344.,  354.,  293.,  269.,  286.,
        738.,  689.,  722.,  684.,  467.,  434.,  493.,  479.,  501.,
        258.,  255.,  244.,  249.,  285.,  253.,  246.,  280.,  241.,
        281.,  235.,  238.,  252.,  362.,  306.,  343.,  310.,  442.,
        229.,  376.,  351.,  328.,  318.,  318.,  307.,  253.,  227.,
        239.,  245.,  341.,  337.,  328.,  311.,  279.,  259.,  277.,
        286.,  376.,  446.,  353.,  360.,  374.,  510.,  490.,  458.,
        425.,  522.,  927.,  555.,  550.,  516.,  548.,  560.,  545.,
        633.,  496.,  498.,  407.,  413.,  383.,  353.,  483.,  421.,
        356.,  366.,  427.,  365.]

const ind = [ 0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,
        3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  6,  6,  6,  6,
        6,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9, 10,
       10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13,
       13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16,
       16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20,
       20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23,
       23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25] .+ 1

logy = log.(y)

const I = length(y)
const J = 26
const zlogy = (logy .- mean(logy)) ./ std(logy)



function logprior_theta(theta::Array{Float64}, mu::Float64, tau::Float64)
    return sum(-log(tau) .- 0.5 * ((theta .- mu) ./ tau).^2)
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
    theta = beta[1:J]
    mu = beta[J + 1]
    sigma = beta[J + 2]
    tau = beta[J + 3]

    lp = 0.0
    for i in 1:I
        j = ind[i] # here I obtain the individual j corresponding to the i:th measurement
        lp += -log(sigma) - 0.5 * ((zlogy[i] - theta[j]) / sigma)^2
    end
    return lp
end

function logpost(beta::Array{Float64})
    theta = beta[1:J]
    mu = beta[J + 1]
    sigma = beta[J + 2]
    tau = beta[J + 3]
    if (sigma > 0) && (tau > 0)
        return logll(beta) + logprior_theta(theta, mu, tau) + logprior_mu(mu) + logprior_sigma(sigma) + logprior_tau(tau)
    else
        return -Inf
    end
end


beta = ones(J + 3)
zeta_samp = 0
if run_mcmc
    N = 1000# + Nb
    #N = 150 + Nb
    #zeta_samp,lp = fss_sampzele_mp(beta, 0.1*ones(length(beta)), logpost; N=N, Nb = Nb,nr_processes=8)
    zeta_samp, lp = slice_sample(beta, ones(length(beta)), logpost, 500; printing=true)
    C = cov(zeta_samp')
    Random.seed!(1)
    zeta_samp, lp = ABDA.fslice_sample(deepcopy(beta), deepcopy(C), logpost, 10; printing=true)
    
    println("\n"^100)
    Random.seed!(1)
    zeta_samp2, lp2 = ABDA.fslice_sample_original(deepcopy(beta), deepcopy(C), logpost, 10; printing=true)


end
error()

figure()
plot(zeta_samp')


figure()
N = Int(ceil(sqrt(length(beta))))
for i in 1:length(beta)
    subplot(N, N, i)
    hist(zeta_samp[i,:], 100)
    xlabel(string("\$\\theta_{", i, "}\$"))
end
tight_layout()

theta_samp = zeta_samp[1:J,:]
mu_samp = zeta_samp[J + 1,:]
sigma_samp = zeta_samp[J + 2,:]
tau_samp = zeta_samp[J + 3,:]

# (ln(y)-my)/sy = theta + s*e
# ln(y) = sy*(theta + s*e) + my
# ln(y) = beta + sigma*e, beta = sy*theta+my, sigma = sy*s
# y = exp(sy*theta + my + sy*s*e)
# y = exp(sy*theta+my)*exp(sy*s*e)
# y = exp(theta2)*exp(sigma2*e)

sigma2_samp = std(logy) .* sigma_samp
theta2_samp = std(logy) .* theta_samp .+ mean(logy)

# mean value of log-normal pdf
# https://en.wikipedia.org/wiki/Log-normal_distribution
mu_y_samp = exp.(theta2_samp .+ repeat(sigma2_samp', size(theta2_samp, 1), 1).^2 ./ 2)



figure(figsize = (8 * 2, 6 * 2), dpi = 80)
N = Int(ceil(sqrt(J)))
col = "b"
for i in 1:J
    subplot(N, N, i)
    x = mu_y_samp[i,:]
    ABDA.hist(x, color = col)
    xlabel(string("\$\\exp(\\theta_{", i, "}+\\sigma^2/2)\$"))
end
tight_layout()

figure()
col = "b"
i = 8
x = mu_y_samp[i,:]
ABDA.hist(x, color = col)
xlabel(string("\$\\exp(\\theta_{", i, "}+\\sigma^2/2)\$"))
title("The expected reaction time for the dude")
tight_layout()



mu2_samp = std(logy) .* mu_samp .+ mean(logy)
tau2_samp = std(logy) .* tau_samp

exp_mu2_samp = exp.(mu2_samp .+ 0.5 .* tau2_samp.^2 .+ 0.5 .* sigma2_samp.^2)
figure()
ABDA.hist(exp_mu2_samp, color = "b")
xlabel(string("\$\\exp(\\mu + \\tau^2/2 + \\sigma^2/2)\$"))
title("expected reaction time for the group")
tight_layout()

my = mean(logy)
sy = std(logy)
y_pred = zeros(size(zeta_samp, 2))
for i in 1:size(zeta_samp, 2)
    theta_i = mu_samp[i] + tau_samp[i] * randn()
    logzy_i = theta_i + sigma_samp[i] * randn()
    logy_i = sy * logzy_i + my
    y_pred[i] = exp(logy_i)
end

figure()
ABDA.hist(y_pred)
title("posterior predicted reaction time a random individual")

figure()
for j in 1:J
    plot(log.(y[ind .== j]))
end

mle = []
for j in 1:J
    push!(mle, mean(logy[ind .== j]))
end

function logprior_thetas(theta, mu, tau) # p(theta[1],...,theta[J]) ~ N(mu,tau)
    return (-log.(tau) .- 0.5 .* ((theta .- mu) ./ tau).^2)
end

offset = 500
figure()
for j in 1:J
#    fr,bins = np.histogram(theta2_samp[j,:],Int(round(sqrt(length(theta2_samp[j,:])))),normed=true)
#    x,y = get_mystep(bins,fr)
#    plot(x,y*0.3+j)
    ABDA.hist(theta2_samp[j,:], baseline = j * offset, color = "b")
    plot(mle[j], j * offset, "ro")
end
plot(mle[1], 1 * offset, "ro", label = "MLE estimate \$\\hat{\\theta}_{\\mathrm{MLE}}\$")
xl = xlim()
thetas = range(xl[1], stop=xl[2], length=1000)
plot(thetas, 10*offset * exp.(logprior_thetas(thetas, mean(mu2_samp), mean(tau2_samp))), label = "\$p(\\hat{\\mu},\\hat{\\tau)}\$", color = "m")
legend()
yticks([])
xlabel(string("\$\\theta_j\$ expected log(reaction time)"))
tight_layout()



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
ABDA.hist(std(logy) * sigma_samp)
xlabel(string("\$\\sigma\$"))
tight_layout()