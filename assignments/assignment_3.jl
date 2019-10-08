using PyPlot
using Random
using Statistics
using Test
using SpecialFunctions

close("all")


if true # this is task A.2
    P = Dict{String,Float64}()

    P[":("] = 0.001
    P[":)"] = 1.0-P[":("]
    
    P["+|:("] = 0.99
    P["+|:)"] = 0.05

    P["-|:("] = 1-0.99
    P["-|:)"] = 1-0.05

    P["+"] = P["+|:)"]*P[":)"] + P["+|:("]*P[":("]
    
    P[":(|+"] = P["+|:("]*P[":("] / P["+"]

    P["-"] = P["-|:)"]*P[":)"] + P["-|:("]*P[":("]
    P[":(|-"] = P["-|:("]*P[":("] / P["-"]
    


    function post1(T) # version 1 just to update likelihood keeping the same prior 
        prior = [P[":("], P[":)"]]
        likelihood = [1.0,1.0]
        for t in T
            likelihood[1] *= P[t*"|:("]
            likelihood[2] *= P[t*"|:)"]
        end
        
        posterior = likelihood.*prior./sum(likelihood.*prior)
        return posterior
    end

    function post2(T) # version 2 is doing sequential Bayes updating the prior using the posterior of the previous test result
        # initialize prior and likelihood and posterior
        prior = [P[":("], P[":)"]]
        likelihood = [1.0,1.0]
        posterior = likelihood.*prior./sum(likelihood.*prior) # Bayes formula with no data posterior = prior
        for t in T # if data in T we will enter the loop
            likelihood[1] = P[t*"|:("] # get likelihood for result in t given sick
            likelihood[2] = P[t*"|:)"] # get likelihood for result in t given healthy
            posterior = likelihood.*prior./sum(likelihood.*prior) # calc posterior
            prior = posterior # update prior until the next experiment
        end
        return posterior
    end

    for T = ["","+","+-","+-++-----+----"]
        println("Testing T=\"$T\"") 
        @test post1(T) ≈ post2(T)
    end
    println("----- Task A.2.a ------")
    for T = ["","+","+-"] 
        println("The posterior for p(θ|T=\"$T\") = $(post2(T))")
    end
end


if true
    
    function bernoulli_pdf(y,θ) # Eq. (6.1)
        return θ^y*(1.0-θ)^(1.0-y)
    end
    
    function likelihood(y,θ) # Eq. (6.2) first row
        return prod(bernoulli_pdf.(y,θ))
    end
    
    function bernoulli_log_pdf(y,θ) # Eq. (6.1)
        return log(θ)*y + log(1.0-θ)*(1.0-y)
    end
    
    function log_likelihood(y,θ) # Eq. (6.2) first row
        return sum(bernoulli_log_pdf.(y,θ))
    end

    println("----- Task B.4.b ------")
    for n in [10,1000,100000]
        y = round.(rand(n))
        println("With size(y)=$n, likelihood: ", likelihood(y,0.5), ", log-likelihood: ", log_likelihood(y,0.5))
    end
    
    figure()
    for y in [[1],[1,1],[1,1,0,1]] 
        θs = 0:0.01:1
        likes = zeros(size(θs))
        for i in 1:length(θs)
            θ = θs[i]
            likes[i] = likelihood(y,θ)
        end
        plot(θs,likes, label=raw"$y = " * "$y" * raw"$")
        
    end
    legend()
    xlabel(raw"$\theta$")
    ylabel(raw"$\mathcal{L}(\theta) = p(y|\theta)$")
    axis("tight")
    grid("on")
    tight_layout()
end



if true
    
    function log_posteior(θ,z::Number,N::Number,a::Number,b::Number) # Eq. (6.8) using a summary of the data z, N
        return (z+a-1)*log(θ) + (N-z+b-1)*log(1-θ) - SpecialFunctions.lbeta(z+a,N-z+b)
    end


    figure()
    for y in [[1],[1,1],[1,1,0,1]] 
        a, b = 1,1
        N = length(y)
        z = sum(y)

        θs = 0:0.001:1
        plot(θs,exp.(log_posteior.(θs,z,N,a,b)), label="\$y=$y\$")
        legend()
        xlabel(raw"$\theta$")
        ylabel(raw"$p(\theta|y)$")
        axis("tight")
        grid("on")
        tight_layout()
    end
end




if true
    

    function log_posteior(θ,y::Vector{Number}) # Eq. (6.2) using data y
        N = length(y)
        z = sum(y)
        return (z+a-1)*log(θ) + (N-z+b-1)*log(1-θ) - SpecialFunctions.lbeta(z+a,N-z+b)
    end

    function log_posteior(θ,z::Number,N::Number,a::Number,b::Number) # Eq. (6.8) using a summary of the data z, N
        return (z+a-1)*log(θ) + (N-z+b-1)*log(1-θ) - SpecialFunctions.lbeta(z+a,N-z+b)
    end


    # From appendix 6.6
    figure()
    for setting = zip([0.75, 0.5, nothing],[25,500,nothing]) # Specify the prior mode.
        t,n = setting
        if t != nothing
            #n = 25 # Specify the effective prior sample size.
            a = t*(n-2) + 1 # Convert to beta shape parameter a.
            b = (1-t)*(n-2) + 1 # Convert to beta shape parameter b.t = 0.75
        else
            a,b = 1,1
        end

        N = 20
        z = 17

        θs = 0:0.001:1
        plot(θs,exp.(log_posteior.(θs,z,N,a,b)), label="\$N=$N\$, \$z=$z\$, \$a=$a\$, \$b=$b\$")
        legend()
        xlabel(raw"$\theta$")
        ylabel(raw"$p(\theta|z,N)$")
        axis("tight")
        grid("on")
        tight_layout()
    end
end



