using LinearAlgebra, Statistics, Random

mutable struct Samples
    thetas::Array{Float64}
    logpdfs::Vector{Float64}
    function Samples(thetas,logpdfs)
        new(thetas,logpdfs)
    end
    function Samples(thetas)
        new(thetas)
    end
    function Samples()
        new()
    end
end



function slice_sample(x0, w, log_pdf, N; m = 1e2, printing = true)
    # https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
    D = length(x0)
    xs = zeros(D,N)
    lp = zeros(N)

    evals = 0
    log_pdf_x1 = 0.0
    for i in 1:N
        if printing && (mod(i,round(N/10))==0)
            print(Int(round(i/N*10)))
        end
        l = 1*x0
        r = 1*x0
        x1 = 1*x0
        for d in randperm(D)
            lu = log(rand())
            u1 = rand()
            v1 = rand()

            if i == 1
                y = log_pdf(x0) + lu
                evals = 1
            else
                y = log_pdf_x1 + lu
                evals = 0
            end

            l[d] = x0[d] - u1*w[d]
            r[d] = l[d] + w[d]

            j = floor(m*v1)
            k = (m-1)-j
            #println(log_pdf(l))
            #println(y)
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

        xs[:,i] = x1
        lp[i] = log_pdf_x1
    end
    return xs, lp
end




function fslice_sample(x0, C, log_pdf, N; m = 1e2, printing = true)
    # https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
    lam,E = eigen(C)
    w = sqrt.(abs.(lam))

    D = length(x0)
    xs = zeros(D,N)
    lp = zeros(N)

    evals = 0
    log_pdf_x1 = 0.0
    wd = zeros(D)



    for i in 1:N
        if printing && (mod(i,round(N/10))==0)
            print(Int(round(i/N*10)))
        end


        l = 1*x0
        r = 1*x0
        x1 = 1*x0
        for d in randperm(D)
            lu = log(rand())
            u1 = rand()
            v1 = rand()

            if i == 1
                y = log_pdf(x0) + lu
                evals = 1
            else
                y = log_pdf_x1 + lu
            end

            wd = w[d]*E[:,d]
            l = x0 - u1*wd
            r = l + wd

            j = floor(m*v1)
            k = (m-1)-j
            while ((y < log_pdf(l)) && (j>0))
                evals += 1
                l -= wd
                j -= 1
            end
            while ((y < log_pdf(r)) && (k>0))
                evals += 1
                r += wd
                k -= 1
            end
            while true
                u2 = rand()
                x1 = l + u2*(r-l)
                log_pdf_x1 = log_pdf(x1)
                # println(y, " ", log_pdf_x1)

                evals += 1
                if (y <= log_pdf_x1)
                    x0 = 1*x1
                    break
                end
                if sign.(r-x0) == sign.(x1-x0)
                    r = 1*x1
                else
                    l = 1*x1
                end
            end

        end

        xs[:,i] = x1
        lp[i] = log_pdf_x1
    end
    return xs, lp
end



function sample(x0, w, log_pdf, N = 10_000, N_burn_in = nothing; m = 1e2, printing = true)
    if N_burn_in == nothing
        N_burn_in = max(round(Int(N*0.1)),100)
    end
    # first run
    xs, lp = slice_sample(x0, w, log_pdf, N_burn_in; m = m, printing = false)
    x0 = xs[:,argmax(lp)]
    w = std(xs,dims=2)
    
    # second run
    xs, lp = slice_sample(x0, w, log_pdf, N_burn_in; m = m, printing = false)


    C = cov(xs')
    x0 = xs[:,argmax(lp)]
    return fslice_sample(x0, C, log_pdf, N; m = m, printing = printing)

end

