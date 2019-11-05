using Random
using Statistics


N = 100_000
tau = 1.5
sigma = 2
mu = 2.5
theta = mu .+ tau*randn(N) # theta ~ N(mu,tau)
x = theta .+ sigma.*randn(N) # x ~ N(theta, sigma)

println("var(x): ",var(x))
println("sigma^2 + tau^2: ",sigma^2 + tau^2)
println("(sigma + tau)^2: ",(sigma + tau)^2)
println("mean(x): ", mean(x))
println("mu: ", mu)

y = exp.(x)
println("mean(y): ", mean(y))
println("E(y): ", exp(mu + (tau^2 + sigma^2)/2))
