using ABDA
using PyPlot
using SpecialFunctions
using Statistics
using Distributions
using Random

# data
string = "Feb. 9
40,553	3,001	8%
Feb. 8
37,552	2,676	8%
Feb. 7
34,876	3,437	11%
Feb. 6
31,439	3,163	11%
Feb. 5
28,276	3,723	15%
Feb. 4
24,553	3,927	19%
Feb. 3
20,626	3,239	19%
Feb. 2
17,387	2,836	19%
Feb. 1
14,551	2,603	22%
Jan. 31
11,948	2,127	22%
Jan. 30
9,821	2,005	26%
Jan. 29
7,816	1,755	29%
Jan. 28
6,061	1,482	32%
Jan. 27
4,579	1,778	63%
Jan. 26
2,801	786	39%
Jan. 25
2,015	703	54%
Jan. 24
1,312	468	55%
Jan. 23
844	265	46%"

strings = split(replace(string,","=>""),"\n")[2:2:end]

dy = [] 
for s in strings
    pushfirst!(dy,parse(Int,split(s,"\t")[2]))
end

y = [] 
for s in strings
    pushfirst!(y,parse(Int,split(s,"\t")[1]))
end

#dy = diff(y)

# https://www.researchgate.net/publication/40823662_Modeling_the_Cumulative_Cases_from_SARS



# log-likelihood Poisson distribution
function log_likelihood(θ,dy)
    α, μ, σ = θ
    t = 1:length(dy)
    λ = α.*(exp.(.-(t .- μ)./σ)./(σ.*(1 .+ exp.(.-(t .- μ)./σ)).^2))
    if any(λ .<= 0)
        return -Inf
    else
        return sum(dy.*log.(λ) .- λ - loggamma.(dy .+ 1)) 
    end
end


function log_likelihood2(θ,y)
    α, μ, σ = θ
    t = 1:length(y)
    λ = α.*(1 ./ (1 .+ exp.(.-(t .- μ)./σ)))
    if any(λ .<= 0)
        return -Inf
    else
        return sum(y.*log.(λ) .- λ - loggamma.(y .+ 1)) 
    end
end

# log prior
function log_prior(θ)
    α, μ, σ = θ
    if α .<= 0.0 || μ .<= 0.0 || σ .<= 0.0 
        return -Inf
    else
        return 0.0
    end
end

# log posterior
log_posterior(θ) = log_likelihood(θ,dy) + log_prior(θ)



## sample the posterior
θ = [5000,15,200.0]
N_burn_in  = 1000
θ_samp, lps = ABDA.sample(copy(θ), ones(length(θ)), log_posterior, 101_000, N_burn_in; printing=true)

# remove "burn in" phase
θ_samp = θ_samp[:,N_burn_in+1:end]
lps = lps[N_burn_in+1:end] 



# plot mcmc chain
for n = 1:3
figure()
subplot(1,2,1), plot(θ_samp[n,:])
subplot(1,2,2), hist(θ_samp[n,:],1000)
tight_layout()
end

t = 1:40
ks = 1:100:size(θ_samp,2)
ns = 1:length(t)
dy_pred = zeros(length(ks),length(ns)) 

for n in ns
    i = 1
    for k in ks
        α, μ, σ = θ_samp[:,k]
        λ = α.*(exp.(.-(t[n] .- μ)./σ)./(σ.*(1 .+ exp.(.-(t[n] .- μ)./σ)).^2))
        dy_pred[i,n] = rand(Poisson(λ))
        i +=1
    end
end
    
y_pred = zeros(length(ks),length(ns)) 
y_pred[:,1] .= y[1] 
for i in 1:length(ks)
    for n in 2:length(t)
        y_pred[i,n] = y_pred[i,n-1] + dy_pred[rand(1:length(ks)),n] 
    end
end
    


ci = Array(ABDA.hdi(y_pred')')
m = mean(y_pred,dims=1)

figure()
subplot(211)
plot(1:length(y),y,"o")
plot(t,m[:],"r-")
plot(t,ci[:,1],"r--")
plot(t,ci[:,2],"r--")
xlabel("days")
ylabel("infected")
grid("on")

ci = Array(ABDA.hdi(dy_pred')')
m = mean(dy_pred,dims=1)

subplot(212)
grid("on")
plot(1:length(dy),dy,"o")
plot(t,m[:],"r-")
plot(t,ci[:,1],"r--")
plot(t,ci[:,2],"r--")
xlabel("days")
ylabel("infected per day")
grid("on")
tight_layout()