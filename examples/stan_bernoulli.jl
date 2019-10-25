# Following https://stanjulia.github.io/CmdStan.jl/stable/WALKTHROUGH/
using CmdStan
using StatsPlots
using MCMCChain

const bernoullistanmodel = "
data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
} 
parameters {
  real<lower=0,upper=1> theta;
} 
model {
  theta ~ beta(1,1);
    y ~ bernoulli(theta);
}
"

stanmodel = Stanmodel(name="bernoulli", model=bernoullistanmodel);

bernoullidata = Dict("N" => 10, "y" => [0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
ProjDir = "."

rc, chns, cnames = stan(stanmodel, bernoullidata, ProjDir, CmdStanDir=CMDSTAN_HOME)


describe(chns)
theta = get_params(chns).theta[:]
plot(chns)
