using ABDA
using PyPlot
using SpecialFunctions
using Statistics
using Distributions
using Random
close("all")



function parse_data(string)
    strings = split(replace(string,","=>""),"\n")[2:2:end]

    y = Float64[] 
    dy = Float64[] 
    for s in strings
        tmp = split(s,"\t")
        if length(tmp)==1
            tmp = split(s," ")
            tmp = [t for t in tmp if t!=""]
        end
        pushfirst!(dy,parse(Float64,tmp[2]))
        pushfirst!(y,parse(Float64,tmp[1]))
    end

    return y, dy
end

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

const y, dy = parse_data(string)

# https://www.researchgate.net/publication/40823662_Modeling_the_Cumulative_Cases_from_SARS
# https://www.maa.org/book/export/html/115630

# dy/dt = r/M*y*(M-y)
# y = M*y0/(y0 + (M-y0)*exp(-r*t))

# The derivative
# dy/dt = -M*y0/(y0 + (M-y0)*exp(-r*t))^2 * (M-y0)*exp(-r*t)*(-r)
# dy/dt = M*y0*(M-y0)*r*exp(-r*t)/(y0 + (M-y0)*exp(-r*t))^2 


# log-likelihood Poisson distribution
function log_likelihood(θ::Vector{Float64},dy::Vector{Float64})
    y0, r, M, a = θ
    t = 1:length(dy)
    λ = M.*y0.*(M .- y0).*r.*exp.( -r.*t) ./ (y0 .+ (M .- y0).*exp.(-r.*t)).^2 
    if any(λ .<= 0) || any(θ .<= 0) || a <= 0.0 || M <= y0 || r >= 1
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
    y0, r, M, a = θ
    if any(θ .<= 0.0) 
        return -Inf
    else
        m,s = 55e3,55e3
        return -0.5*((M-m)/s).^2
    end
end


# log posterior
log_posterior(θ::Vector{Float64}) = log_likelihood(θ,dy) + log_prior(θ)



## sample the posterior
θ = [y[1]-dy[1], 0.2, 60_000.0, 0.1]
N_burn_in  = 10_000
Random.seed!(1)
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
        y0, r, M, a = θ_samp[:,k]
        λ = M.*y0.*(M .- y0).*r.*exp.( -r.*t[n]) ./ (y0 .+ (M .- y0).*exp.(-r.*t[n])).^2 
        r,p = λa2rp(λ,a)
        dy_pred[i,n] = rand(NegativeBinomial(r,1-p))
        i +=1
    end
end


y_rep = zeros(length(ks),length(ns)) 
y_rep[:,1] .= y[1] 
for i in 1:length(ks)
    for n in 2:length(t)
        y_rep[i,n] = y_rep[i,n-1] + dy_pred[rand(1:length(ks)),n] 
    end
end


y_0 = y[1]- dy[1]
y_pred = zeros(length(ks),length(ns)) 
n0 = length(dy)
for i in 1:length(ks)
    y_pred[i,1] = y_0 + dy_pred[rand(1:length(ks)),1]
    for n in 2:length(t)
        if n<=n0
            samp = 0.0
            while true
                samp = y_pred[i,n-1] + dy_pred[rand(1:length(ks)),n]
                if samp >= y[n]
                    break
                end
            end
            y_pred[i,n] = samp
        else
            y_pred[i,n] = y_pred[i,n-1] + dy_pred[rand(1:length(ks)),n] 
        end
    end
end




ABDA.hdi(θ_samp)

string = raw"Feb. 11
45,170	2,071	5%
Feb. 10
43,099	2,546	6%
Feb. 9
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
24,553	3,925	19%
Feb. 3
20,628	3,239	19%
Feb. 2
17,389	2,837	19%
Feb. 1
14,552	2,604	22%
Jan. 31
11,948	2,127	22%
Jan. 30
9,821	2,008	26%
Jan. 29
7,813	1,755	29%
Jan. 28
6,058	1,477	32%
Jan. 27
4,581	1,781	64%
Jan. 26
2,800	785	39%
Jan. 25
2,015	698	53%
Jan. 24
1,317	472	56%
Jan. 23
845	266	46%"

y_new, dy_new = parse_data(string)









ci_rep = Array(ABDA.hdi(y_rep')')
m_rep = mean(y_rep,dims=1)

ci_pred = Array(ABDA.hdi(y_pred')')
m_pred = mean(y_pred,dims=1)

figure()
subplot(311)
plot(1:length(y),y,"ko")
ind = length(y)+1:length(y_new)
plot(ind,y_new[ind],"rs")

plot(t,m_rep[:],"k-", label="prediction")
fill_between(t,ci_rep[:,1],ci_rep[:,2],color="k", alpha=0.2)
#plot(t,ci_rep[:,1],"k--")
#plot(t,ci_rep[:,2],"k--")

plot(t,m_pred[:],"r-", label="step ahead prediction")
fill_between(t,ci_pred[:,1],ci_pred[:,2],color="r", alpha=0.2)
#plot(t,ci_pred[:,1],"b--")
#plot(t,ci_pred[:,2],"b--")
legend()
xlabel("days")
ylabel("infected")
grid("on")


ci = Array(ABDA.hdi(dy_pred')')
m = mean(dy_pred,dims=1)

subplot(312)
grid("on")
plot(1:length(dy),dy,"ko")
plot(ind,dy_new[ind],"rs")
plot(t,m[:],"k-")
plot(t,ci[:,1],"k--")
plot(t,ci[:,2],"k--")
xlabel("days")
ylabel("infected per day")
grid("on")

subplot(313)
ABDA.hist(θ_samp[3,:].*1e-3,40:0.2:80,color="k")
xlim([40, 80])
xlabel(raw"total infected ($\times 1000$)")
yticks([])

tight_layout()


figure()
ABDA.hist(θ_samp[2,:],color="k")
xlabel("rate of growth")
yticks([])




figure()
ABDA.hist(θ_samp[2,:],color="k")
xlabel("rate of growth")
yticks([])

