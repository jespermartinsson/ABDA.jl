
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




# this is the original slice sample from referece 
# https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
function slice_sample_original(x0, w, log_pdf, N; m = 1e2, printing = false)
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



# Take a single slice 
function slice!(x0, l,r,x1, log_pdf, log_pdf_x1, w; m=1e2)
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
    return log_pdf_x1
end

# https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
function slice_sample(x0, w, log_pdf, N; m = 1e2, printing = false)
    D = length(x0)
    xs = zeros(D,N)
    lp = zeros(N)
    
    # pre-allocate
    l = similar(x0)
    r = similar(x0)
    x1 = similar(x0)

    evals = 0
    log_pdf_x1 = log_pdf(x0)
    for i in 1:N
        if printing && (mod(i,round(N/10))==0)
            print(Int(round(i/N*10)))
        end
        l .= x0
        r .= x0
        x1 .= x0
        log_pdf_x1 = slice!(x0,l,r,x1,log_pdf,log_pdf_x1,w;m=m)
        xs[:,i], lp[i] = x0, log_pdf_x1
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



# try eliptical slice sample https://arxiv.org/abs/1001.0175
# https://www.youtube.com/watch?v=HfzyuD9_gmk
function elip_sample()
    return 
end


function hdi(theta_samp,alpha=0.05)
    cred_mass = 1.0-alpha
    ci = zeros(2)
    if length(size(theta_samp))>1
        K,N = size(theta_samp)
        cis = zeros(2,K)
        for k in 1:K

            ts = theta_samp[k,:]
            sind = sortperm(ts)
            sts = ts[sind]

            N = length(sind)
            length_ci = Inf
            for i in 1:Int(floor(N*alpha))
                i2 = Int(floor(N*cred_mass)+i)
                prop_ci = [sts[i],sts[i2]]
                length_prop_ci = prop_ci[2]-prop_ci[1]
                if length_prop_ci < length_ci
                    ci = prop_ci
                    length_ci = ci[2]-ci[1]
                end
            end
            cis[:,k] = ci

        end
        return cis
    else
        N = length(theta_samp)

        ts = theta_samp
        sind = sortperm(ts)
        sts = ts[sind]

        N = length(sind)
        length_ci = Inf
        for i in 1:Int(floor(N*alpha))
            i2 = Int(floor(N*cred_mass)+i)
            prop_ci = [sts[i],sts[i2]]
            length_prop_ci = prop_ci[2]-prop_ci[1]
            if length_prop_ci < length_ci
                ci = prop_ci
                length_ci = ci[2]-ci[1]
            end
        end
        return ci
    end
end








function mystep(x,y;color="k",width=1,label="",alpha=1)
    for n in 1:(length(y)-1)
        dx = x[n+1]-x[n]
        xv = x[n] + dx*0.5 + [-1,1,1]*dx*width*0.5
        yv = [y[n],y[n],y[n+1]]
        if (n==1) & (label != "")
            plot(xv,yv,color=color,label=label,alpha=alpha)
        else
            plot(xv,yv,color=color,alpha=alpha)
        end
    end
end

function get_mystep(x,y;color="k",width=1,label="",alpha=1)
    xv = []
    yv = []
    for n in 1:(length(y)-1)
        push!(xv, x[n])
        push!(xv, x[n+1])
        push!(yv,y[n])
        push!(yv,y[n])
    end
    return xv,yv
end

# https://en.wikipedia.org/wiki/Correlation_coefficient
function acov(x,k=0)
  zx = x .- mean(x)
  zxk = zx[k+1:end]
  zyk = zx[1:end-k]
  return sum(zxk.*zyk)/sqrt(sum(zxk.^2)*sum(zxk.^2))
end

function acovlim(x;lim=0.05)
  k = 0
  rhos = []
  rho = 1
  while rho>lim
    rho = acov(x,k)
    push!(rhos,rho)
    k += 1
  end
  return rhos
end

# ess -- effective sample size (Kruschke 2014, page 184)
function ess(x)
    if typeof(x)==Vector{Float64}
        n = length(x)
        acf = acovlim(x)
        return n/(1+2*sum(acf[2:end]))
    else
        m,n = size(x)
        list = zeros(m)
        for i in 1:m
            acf = acovlim(x[i,:])
            list[i] = n/(1+2*sum(acf[2:end]))
        end
        return list
    end
end

# mcse -- monte carlo standard error (Kruschke 2014, page 187)
# The MCSE indicates the estimated SD of the sample mean in the chain,
# on the scale of the parameter value. In Figure 7.11, for example, despite the small ESS,
# the mean of the posterior appears to be estimated very stably.
function mcse(x)
  return std(x,2)./ess(x)
end


function hist(x,bins=0;color = "k",baseline = 0, label=nothing)
    if bins == 0
        bins=Int(round(sqrt(length(x))))
    end
    ci = hdi(x)
    #fr, bins = np.histogram(x,bins,normed = true)
    #fr, bins = PyPlot.hist(x,bins;density = true, show=false)
    hist_data = fit(Histogram,x,nbins=bins)
    fr,bins = hist_data.weights, hist_data.edges[1]
    #plt["hist"](mu_y_samp[i,:],100,alpha=0.5)
    xv,yv = get_mystep(bins,fr)
    fill_between(xv,yv .+ baseline, baseline, color=color,alpha=0.25,label=label)

    ind = (xv.>ci[1]).*(xv.<ci[2])
    fill_between(xv[ind],yv[ind] .+ baseline, baseline, color=color,alpha=0.25)
    plot(ci[[1,1]],[0,yv[ind][1]] .+ baseline,color*"--")
    plot(ci[[end,end]],[0,yv[ind][end]] .+ baseline,color*"--")

    text(ci[1],yv[ind][1] .+ baseline,@sprintf(" %.3g",ci[1]),color=color,rotation=90,va="bottom",ha="center",alpha=0.95)
    text(ci[2],yv[ind][end] .+ baseline,@sprintf(" %.3g",ci[2]),color=color,rotation=90,va="bottom",ha="center",alpha=0.95)

    mx = mean(x)
    ind = (xv.<mx)
    plot(mx*ones(2),[0,yv[ind][end]] .+ baseline,color*"--")
    text(mx,yv[ind][end] .+ baseline,@sprintf(" %.3g",mx),color=color,rotation=90,va="bottom",ha="center",alpha=0.95)
end