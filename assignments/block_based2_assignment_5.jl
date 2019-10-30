#using ABDA
using PyPlot
using Random





# Take a single slice 
function slice!(x0, l, r, x1, log_pdf, log_pdf_x1, w; m=1e2)
    D = length(x0)
    for d in randperm(D)
        lu = log(rand())
        u1 = rand()
        v1 = rand()

        y = log_pdf_x1 + lu
        evals = 0

        l[d] = x0[d] - u1*w[d]
        r[d] = l[d] + w[d]

        j = floor(m*v1)
        k = (m-1)-j

        while ((y < log_pdf(l)) && (j>0))
            evals += 1
            l[d] -= w[d]
            j -= 1
        end
        while ((y < log_pdf(r)) && (k>0))
            evals += 1
            r[d] += w[d]
            k -= 1
        end
        while true
            u2 = rand()
            x1[d] = l[d] + u2*(r[d]-l[d])
            log_pdf_x1 = log_pdf(x1)
            evals += 1
            if (y <= log_pdf_x1)
                x0[d] = x1[d]
                break
            elseif (x1[d]<x0[d])
                l[d] = x1[d]
            elseif (x1[d]>x0[d])
                r[d] = x1[d]
            else
                throw(ErrorException("shrinkage error"))
            end
        end
    end
    return log_pdf_x1
end

# https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
function block_slice_sample(x0, w, log_pdf, N; m = 1e2, printing = false)
    # number of blocks
    B = length(x0)

    # initiate chain
    xs = []
    for b in 1:B
        push!(xs,Array{Float64,2}(undef,(length(θs[b]),N)))
    end 

    lp = zeros(N)
    
    evals = 0
    log_pdf_x1 = log_pdf(x0)
    lp[1] = log_pdf_x1 
    for i in 2:N
        lp[i] = lp[i-1]
        if printing && (mod(i,round(N/10))==0)
            print(Int(round(i/N*10)))
        end
        l = 1*x0
        r = 1*x0
        x1 = 1*x0
        for b in 1:B
            log_pdf_x1_b = log_pdf(x1[b],b)
            lp[i] -= log_pdf_x1_b
            log_pdf_x1_b = slice!(x0[b], l[b], r[b], x1[b], (x)->log_pdf(x,b), log_pdf_x1_b, w[b]; m=m)
            xs[b][:,i] .= x0[b]
            lp[i] += log_pdf_x1_b
        end
    end
    return xs, lp
end






close("all")

include("parser_5.jl")

# read data from csv
filename = (@__DIR__) * raw"/data/ABDA 2019 -- Reaction time - Sheet1.tsv"
skip_lines = [14,15,24,25,26,27,42] .- 1 
y, subject = parse_data_array(filename, skip_lines)

B = length(y)
for b in 1:B
    y[b] = log.(float.(y[b]))
end

figure()
for j in 1:length(y)
    i = 1:length(y[j])
    plot(i, y[j], ".-", alpha = .5)
end



mutable struct Likelihood # individual likelihood
    y::Vector{Int64}


    
    # constructors
    Likelihood() = new()
    Likelihood(y) = new(y)
end
function log_pdf(l::Likelihood, θ::Vector{Float64})
    if any(θ .<= 0)
        return -Inf
    else
        ε = l.y .- θ[1]
        return -length(ε)*log(θ[end]) - 0.5*ε'ε/θ[end]^2
    end
end




mutable struct Likelihoods # all likelihoods
    lhs::Vector{Likelihood}

    # constructors
    Likelihoods() = new()
    function Likelihoods(y)
        J = length(y)
        lhs = Vector{Likelihood}(undef,J)
        for j in 1:J
            lhs[j] = Likelihood(y[j])
        end
        new(lhs)
    end
end
function log_pdf(l::Likelihoods, θs::Vector{Vector{Float64}})
    value = 0.0
    for j = 1:length(l.lhs)
        value += log_pdf(l.lhs[j], θs[j])
    end
    return value
end

function log_pdf(l::Likelihoods, θ::Vector{Float64},j::Int64)
    return log_pdf(l.lhs[j], θ) 
end


J = length(y)
lhs = Likelihoods(y)
log_pdf(lhs.lhs[1],rand(2))
θs = Vector{Vector{Float64}}(undef,J)
w = Vector{Vector{Float64}}(undef,J)
for j in 1:J
    θs[j] = rand(2)
    w[j] = ones(2)
end
log_pdf(lhs,θs)

log_pdf2(θs) = log_pdf(lhs,θs)
log_pdf2(θ::Vector{Float64}, j::Int64) = log_pdf(lhs,θ,j)
Random.seed!(1)
xs, lp = block_slice_sample(θs, w,  log_pdf2, 10_000; printing=true)

b = 20
figure()
plot(xs[b][1,:],".-")
plot(xs[b][2,:],".-")