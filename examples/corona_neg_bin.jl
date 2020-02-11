using ABDA
using PyPlot
using SpecialFunctions
using Statistics
using Distributions
using Random
close("all")
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

dy0 = Float64[] 
for s in strings
    pushfirst!(dy0,parse(Float64,split(s,"\t")[2]))
end
const dy = dy0

y0 = Float64[] 
for s in strings
    pushfirst!(y0,parse(Float64,split(s,"\t")[1]))
end
const y = y0

#dy = diff(y)

# https://www.researchgate.net/publication/40823662_Modeling_the_Cumulative_Cases_from_SARS



# log-likelihood Poisson distribution
function log_likelihood(θ::Vector{Float64},dy::Vector{Float64})
    α, μ, σ, a = θ
    t = 1:length(dy)
    λ = α.*(exp.(.-(t .- μ)./σ)./(σ.*(1 .+ exp.(.-(t .- μ)./σ)).^2))
    if any(λ .<= 0) || any(θ .<= 0) || a <= 0.0
        return -Inf
    else
        #return sum(loggamma.(dy .+ 1 ./ a) .- (loggamma.(dy .+ 1) .- loggamma.(1 ./ a)) .+ dy.*(log.(a .* λ) .- log.(1 .+ a .* λ)) .+ (-1 ./ a).*log.(1 .+ a .* λ)) 
        return sum(loggamma.(dy .+ 1 ./ a) .- loggamma.(1 ./ a) .- loggamma.(dy .+ 1) .- (1 ./ a).*log.(1 .+ a .* λ) .- dy.*log.(1 .+ a.*λ) .+ dy.*log.(a) .+ dy.*log.(λ)) 
    end
end


function rp2λa(r,p)
    λ =  p*r/(1-p)
    a = 1/(λ*(1-p)) - 1/λ 
    return λ, a
end

function λa2rp(λ,a)
    p = 1-1/λ/(a + 1/λ)   
    r = λ*(1-p)/p 
    return r, p 
end

# log prior
function log_prior(θ::Vector{Float64})
    α, μ, σ, a = θ
    if α .<= 0.0 || μ .<= 0.0 || σ .<= 0.0 || a <= 0.0 
        return -Inf
    else
        m,s = 55e3,55e3
        return -0.5*((α-m)/s).^2
    end
end

# log posterior
log_posterior(θ::Vector{Float64}) = log_likelihood(θ,dy) + log_prior(θ)



## sample the posterior
θ = [55_000,14,3.5,100.5]
N_burn_in  = 10_000
θ_samp, lps = ABDA.sample(copy(θ), 0.3*abs.(θ), log_posterior, 1_010_000, N_burn_in; printing=true)
#θ_samp, lps = ABDA.slice_sample(copy(θ), 0.1*ones(length(θ)), log_posterior, 20_000; printing=true)

# remove "burn in" phase
θ_samp = θ_samp[:,N_burn_in+1:end]
lps = lps[N_burn_in+1:end] 



# plot mcmc chain
for n = 1:4
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
        α, μ, σ, a = θ_samp[:,k]
        λ = α.*(exp.(.-(t[n] .- μ)./σ)./(σ.*(1 .+ exp.(.-(t[n] .- μ)./σ)).^2))
        r,p = λa2rp(λ,a)
        dy_pred[i,n] = rand(NegativeBinomial(r,1-p))
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
    
ABDA.hdi(θ_samp)


ci = Array(ABDA.hdi(y_pred')')
m = mean(y_pred,dims=1)

figure()
subplot(311)
plot(1:length(y),y,"o")
plot(t,m[:],"r-")
plot(t,ci[:,1],"r--")
plot(t,ci[:,2],"r--")
xlabel("days")
ylabel("infected")
grid("on")

ci = Array(ABDA.hdi(dy_pred')')
m = mean(dy_pred,dims=1)

subplot(312)
grid("on")
plot(1:length(dy),dy,"o")
plot(t,m[:],"r-")
plot(t,ci[:,1],"r--")
plot(t,ci[:,2],"r--")
xlabel("days")
ylabel("infected per day")
grid("on")

subplot(313)
ABDA.hist(θ_samp[1,:],40e3:2e2:80e3,color="r")
xlim([40e3, 80e3])
xlabel("total infected")
yticks([])

tight_layout()


figure()
ABDA.hist(1 ./ θ_samp[3,:],color="r")
xlabel("rate of growth")
yticks([])

