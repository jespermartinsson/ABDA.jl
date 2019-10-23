using PyPlot, Random 

# This is the original slice sample from referece 
# https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
function slice_sample(x0, w, log_pdf, N; m = 1e2, printing = false)
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

# metropolis mcmc
function m_sample(x0, sigma, log_pdf, N) 
    D = length(x0)

    x = zeros(D,N)
    lp = zeros(N)
    acc = 0
    xt = zeros(D)
    xp = zeros(D)


    logu_samples = log.(rand(N)) # sample uniform
    prop_samples = sigma*randn(D,N) # sample proposal N(0,sigma). Use sigma = sqrtm(C) if multivariate, where C is the covariace matrix

    for k in 1:D
        x[k,1] = x0[k]
    end
    # initi the current pdf value for xt for t=0
    log_pdf_xt = log_pdf(x0)
    lp[1] = log_pdf_xt
    for t in 1:N-1
        xt = x[:,t]
        xp = xt .+ prop_samples[:,t]

        log_pdf_xp = log_pdf(xp) # calc the proposal
        r = log_pdf_xp - log_pdf_xt # calculate the M ratio

        if logu_samples[t] < r # accept it
            x[:,t+1] = xp
            log_pdf_xt = log_pdf_xp # the old is the new
            acc += 1
        else
            x[:,t+1] = xt
        end
        lp[t+1] = log_pdf_xt
    end
    acceptance = acc/N
    println("acceptance: $acceptance")
    return x, lp
end



close("all")

# define what examples to run
example = ["1D", "2D"]


if any(example .== "1D")
    # create a strange pdf
    function log_pdf_1D(x)
        if -2*pi < x < 2*pi
            return log(sin(x)^2/(1+x^2))
        else
            return -Inf
        end
    end

    # My samplers works only in multi dimensions and the pdf must be defined with an array as input parameter.
    # For the 1D special case the input must also be an array with a single element x[1]. So we do the trick below:
    log_pdf(x) = log_pdf_1D(x[1]) # define log_pdf with array as input. 


    N = 10_000 # number of mcmc samples
    x0 = [0.2] # init guess

    # try metropolis
    sigma = [10.0] # for the proposal distribution. Use sigma = sqrtm(C) if multivariate 
    thetas_m, log_pdfs_m = m_sample(x0, sigma, log_pdf, N)

    # try slice
    w = [10.0] # stepping out value
    thetas_s, log_pdfs_s = slice_sample(x0, w, log_pdf, N)

    N2 = 500 # show only N2 samples in chain
    xl = [-2*pi, 2*pi] # x limits
    figure()
    subplot(221), plot(thetas_m[1:N2],1:N2,label="metropolis"), xlim(xl)
    subplot(222), plot(thetas_s[1:N2],1:N2,label="slice"), xlim(xl)
    subplot(223), hist(thetas_m[:],Int(round(sqrt(N)))), xlim(xl), xlabel(raw"$\theta_{\mathrm{metropolis}}$")
    subplot(224), hist(thetas_s[:],Int(round(sqrt(N)))), xlim(xl), xlabel(raw"$\theta_{\mathrm{slice}}$")

end

if any(example .== "2D")
    
    using LinearAlgebra
    # create a strange pdf
    function log_pdf(x)
        if all(abs.(x) .< 2*pi)
            return sum(log.(sin.(x).^2 ./(1 .+x.^2)))
        else
            return -Inf
        end
    end

 

    N = 10_000 # number of mcmc samples
    x0 = [0.2, -0.1] # init guess

    # try metropolis
    sigma = 10.0*I(2) # for the proposal distribution. Use sigma = sqrtm(C) if multivariate 
    thetas_m, log_pdfs_m = m_sample(x0, sigma, log_pdf, N)

    # try slice
    w = [10.0, 10.0] # stepping out value
    thetas_s, log_pdfs_s = slice_sample(x0, w, log_pdf, N)


    N2 = 500 # show only N2 samples in chain
    xl = [-2*pi, 2*pi] # x limits
    figure()
    subplot(321), plot(thetas_m[1,1:N],thetas_m[2,1:N],".-",alpha=0.3), xlim(xl), ylim(xl)
    subplot(322), plot(thetas_s[1,1:N],thetas_s[2,1:N],".-",alpha=0.3), xlim(xl), ylim(xl)
    subplot(323), plot(thetas_m[1,1:N2],1:N2), xlim(xl)
    subplot(323), plot(thetas_m[2,1:N2],1:N2), xlim(xl)
    subplot(324), plot(thetas_s[1,1:N2],1:N2), xlim(xl)
    subplot(324), plot(thetas_s[2,1:N2],1:N2), xlim(xl)
    subplot(325), hist(thetas_m[1,:],Int(round(sqrt(N))),alpha=0.6), xlim(xl), xlabel(raw"$\theta_{\mathrm{metropolis}}$")
    subplot(325), hist(thetas_m[2,:],Int(round(sqrt(N))),alpha=0.6), xlim(xl), xlabel(raw"$\theta_{\mathrm{metropolis}}$")
    subplot(326), hist(thetas_s[1,:],Int(round(sqrt(N))),alpha=0.6), xlim(xl), xlabel(raw"$\theta_{\mathrm{slice}}$")
    subplot(326), hist(thetas_s[2,:],Int(round(sqrt(N))),alpha=0.6), xlim(xl), xlabel(raw"$\theta_{\mathrm{slice}}$")

end


