#using ABDA
using PyPlot
using Random





# Take a single slice 
function slice!(x0, l,r,x1, log_pdf, log_pdf_x1, w; m=1e2)
    B = length(x0)
    for b in randperm(B)
        D = length(x0[b])
        for d in randperm(D)
            lu = log(rand())
            u1 = rand()
            v1 = rand()

            y = log_pdf_x1 + lu
            evals = 0

            l[b][d] = x0[b][d] - u1*w[b][d]
            r[b][d] = l[b][d] + w[b][d]

            j = floor(m*v1)
            k = (m-1)-j
            #println(log_pdf(l))
            #println(y)
            while ((y < log_pdf(l,b)) && (j>0))
                evals += 1
                l[b][d] -= w[b][d]
                j -= 1
            end
            while ((y < log_pdf(r,b)) && (k>0))
                evals += 1
                r[b][d] += w[b][d]
                k -= 1
            end
            while true
                u2 = rand()
                x1[b][d] = l[b][d] + u2*(r[b][d]-l[b][d])

                println("log_pdf: ", log_pdf_x1)
                log_pdf_x1 = log_pdf(x1,b)
                
                println("log_pdf: ", log_pdf_x1)
                println("y: ", y)
                evals += 1
                if (y <= log_pdf_x1)
                    x0[b][d] = x1[b][d]
                    break
                elseif (x1[b][d]<x0[b][d])
                    l[b][d] = x1[b][d]
                elseif (x1[b][d]>x0[b][d])
                    r[b][d] = x1[b][d]
                else
                    throw(ErrorException("shrinkage error"))
                end
            end
        end
    end
    return log_pdf_x1
end

# https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
function slice_sample(x0, w, log_pdf, N; m = 1e2, printing = false)
    D = length(x0)
    xs = zeros(D,N)
    lp = zeros(N)
    
    evals = 0
    log_pdf_x1 = log_pdf(x0)
    for i in 1:N
        if printing && (mod(i,round(N/10))==0)
            print(Int(round(i/N*10)))
        end
        l = 1*x0
        r = 1*x0
        x1 = 1*x0
        log_pdf_x1 = slice!(x0,l,r,x1,log_pdf,log_pdf_x1,w;m=m)
        xs[:,i], lp[i] = x0, log_pdf_x1
    end
    return xs, lp
end






close("all")

include("parser_5.jl")

# read data from csv
filename = (@__DIR__) * raw"/data/ABDA 2019 -- Reaction time - Sheet1.tsv"
skip_lines = [14,15,24,25,26,27,42] .- 1 
y, subject = parse_data_array(filename, skip_lines)



figure()
for j in 1:length(y)
    i = 1:length(y[j])
    plot(i, y[j], ".-", alpha = .5)
end



mutable struct Likelihood # individual likelihood
    y::Vector{Int64}
    value::Float64

    # constructors
    Likelihood() = new()
    Likelihood(y) = new(y)
end
function log_pdf(l::Likelihood, θ::Vector{Float64})
    if θ[end] <= 0
        l.value = -Inf
    else
        ε = l.y .- θ[1]
        l.value = -length(ε)*log(θ[end]) - 0.5*ε'ε/θ[end]^2
    end
    return l.value
end




mutable struct Likelihoods # all likelihoods
    lhs::Vector{Likelihood}
    value::Float64

    # constructors
    Likelihoods() = new()
    function Likelihoods(y)
        J = length(y)
        lhs = Vector{Likelihood}(undef,J)
        for j in 1:J
            lhs[j] = Likelihood(y[j])
        end
        new(lhs, NaN)
    end
end
function log_pdf(l::Likelihoods, θs::Vector{Vector{Float64}})
    l.value = 0.0
    for j = 1:length(l.lhs)
        l.value += log_pdf(l.lhs[j], θs[j])
    end
    return l.value
end

function log_pdf(l::Likelihoods, θs::Vector{Vector{Float64}},j)
    l.value += -l.lhs[j].value + log_pdf(l.lhs[j], θs[j]) 
    return l.value
end

#log_pdf(l::Likelihoods, θs::Vector{Vector{Float64}},j) = log_pdf(l::Likelihoods, θs::Vector{Vector{Float64}})

function log_pdf(l::Likelihoods, θ::Vector{Float64},j)
    l.value += -l.lhs[j].value + log_pdf(l.lhs[j], θ) 
    return l.value
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
log_pdf2(θs,j) = log_pdf(lhs,θs,j)
slice_sample(θs, w,  log_pdf2, 1)
