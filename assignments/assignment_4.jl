using ABDA
using PyPlot
close("all")

# data
y = [1,0,1,1,0,1,1,1,0,1,1,1,1,1]
z = [1,0,0,0,0,0,0,1,1,0]

# log-likelihood bernoulli
function log_likelihood(θ::Float64,y::Union{Vector{Float64},Vector{Int64}})
    if (0<θ<1)
        n = length(y)
        llh = 0.0
        for i in 1:n
            llh += y[i]*log(θ) + (1-y[i])*log(1-θ)
        end
        return llh
    else
        return -Inf
    end
    
end

# log prior
function log_beta(θ,α,β)
    if 0<θ<1
        return (α-1)*log(θ) + (β-1)*log(1-θ)
    else
        return -Inf
    end
end

# log posterior
log_posterior_y(θ) = log_likelihood(θ[1],y) + log_beta(θ[1],1,1)
log_posterior_z(θ) = log_likelihood(θ[1],z) + log_beta(θ[1],1,1)



## sample the posterior
θ_init = [0.5]
N_burn_in  = 500
θy_samp, lpsy = sample(copy(θ_init), ones(length(θ_init)), log_posterior_y, 1_000_000, N_burn_in)
θz_samp, lpsz = sample(copy(θ_init), ones(length(θ_init)), log_posterior_z, 1_000_000, N_burn_in)

# plot mcmc chain
if false
    figure()
    subplot(1,2,1), plot(θ_samp[1,:])
    subplot(1,2,2), hist(θ_samp[1,:],100)
    tight_layout()

else
    pr1 = sum(θy_samp[1,:] .> 0.5)/size(θy_samp,2)
    pr2 = sum(θy_samp[1,:] .> θz_samp[1,:])/size(θy_samp,2)

    figure()
    N2 = 1000
    subplot(3,1,1), plot(θy_samp[1,1:N2],1:N2,".r-",alpha = 0.25, label = raw"$\theta^{\{i\}}|y$")
    subplot(3,1,1), plot(θz_samp[1,1:N2],1:N2,".b-",alpha = 0.25, label = raw"$\theta^{\{i\}}|z$")
    title("N: $(size(θy_samp,2)), ESS: $(ABDA.ess(θy_samp[1,:]))")
    ylabel(raw"$i$ (sample index)")
    xlabel(raw"$\theta^{\{i\}}$")
    legend()
    subplot(3,1,2), ABDA.hist(θy_samp[1,:],100; color = "r", label=raw"$p(\theta|y)$")
    subplot(3,1,2), ABDA.hist(θz_samp[1,:],100; color = "b", label=raw"$p(\theta|z)$")
    title(raw"$Pr\{\theta_y>\theta_z\}$: "*"$(pr1)")
    ylabel(raw"posterior")
    xlabel(raw"$\theta$")
    legend()
    subplot(3,1,3), ABDA.hist(θy_samp[1,:] .- θz_samp[1,:],100; color = "b", label=raw"$p(\delta\theta|y,z)$")
    title(raw"$Pr\{\theta_y>\theta_z\}$: "*"$(pr2)")
    ylabel(raw"posterior")
    xlabel(raw"$\delta\theta$")
    legend()
    tight_layout()
end

