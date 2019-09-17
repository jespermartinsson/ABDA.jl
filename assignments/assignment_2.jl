using PyPlot
using Random
using Statistics

close("all")

if true # this is Task A:1
    Random.seed!(0)    
    function coin_tosses(theta,n)
        u = rand(n)
        c = zeros(n)
        c[u .< theta] .= 1.0
        return c
    end

    n = 1000
    theta = 0.5
    c = coin_tosses(theta,n)    

    nr_heads = 0
    prop_heads = zeros(n)
    for i in 1:n
        global nr_heads += c[i]
        prop_heads[i] = nr_heads/(i+1)
    end

    figure()
    semilogx(prop_heads,"bo-")
    plot(1:n,theta*ones(n),"k:")
    title("Task A.1.a: recreating Figure 4.1")
    
    
    n = 100
    theta = 0.25
    c = coin_tosses(theta,n)     
    
    figure()
    hist(c,[-0.5,0.5,1.5],normed=true,label="Histogram from n=$n samples")
    stem([0,1],[1-theta,theta],"k",label="True PMF")
    legend()
    title("Task A.1.c")
    
end





if true
    n = 10000
    mu = 3.4
    sigma = sqrt(3)
    x = mu .+ sigma.*randn(n)

        
    
    function normal_pdf(x,mu,sigma) # define a normal probabolity distribution
        y = 1/sqrt(2*pi*sigma^2)*exp(-0.5*(x-mu)^2/sigma^2)
        return y
    end
    
    bins = mu-4*sigma:0.1:mu+4*sigma
    figure()
    hist(x,bins,normed=true,label="Histogram from n=$n samples")


    x_values = mu-4*sigma:0.1:mu+4*sigma
    plot(x_values,normal_pdf.(x_values,mu,sigma),"k-",label=raw"$\mathrm{N}(\mu=3.4,\sigma=\sqrt{3})$")
    leg = legend(fancybox=true)
    leg[:get_frame]()[:set_alpha](0.5)
    title("Task A.2.a")    
    
    dx = 0.1
    x_values = mu-6*sigma:dx:mu+6*sigma
    println("Task A.2.b_________________________")
    println("sample mean = " * string(mean(x)))
    println("Riemann sum and Eq. (4.6) = " * string(sum(x_values.*normal_pdf.(x_values,mu,sigma)*dx)))
    println("True mu = " * string(mu))
    
    println("Task A.2.c_________________________")
    println("sample variace = " * string(var(x)))
    int_mu = sum(x_values.*normal_pdf.(x_values,mu,sigma)*dx)
    println("Riemann sum and Eq. (4.8) = " * string(sum((x_values .- int_mu).^2 .* normal_pdf.(x_values,mu,sigma)*dx)))
    println("True mu = " * string(sigma^2))
    
    

    n = 10000
    mu = 0
    sigma = 1
    x = mu .+ sigma.*randn(n)
    y = exp.(x)

    function lognormal_pdf(x,mu,sigma)
        if x > 0
            y = 1.0/x/sqrt(2*pi*sigma^2)*exp(-0.5*(log(x)-mu)^2/sigma^2)
        else
            y = Inf
        end
        
        return y    
    end

    bins = 0:0.1:10 .- 0.05
    figure()
    hist(y,bins,normed=true,label="Histogram from n=$n samples")

    x_values = 0:0.01:10
    plot(x_values,lognormal_pdf.(x_values,mu,sigma),"k-",label=raw"$\mathrm{Lognormal}(\mu=0,\sigma=1)$")
    leg = legend(fancybox=true)
    leg[:get_frame]()[:set_alpha](0.5)
    title("Task A.2.d.ii")        

    z = lognormal_pdf.(y,mu,sigma)
    sample_mode = y[argmax(z)]
    println("Task A.2.d.iii_____________________")
    println("sample mode = " * string(sample_mode))
    
    
    
    using Optim

    initial_guess = [2.0]
    function neg_lognormal_pdf(x)
        return -lognormal_pdf.(x,0,1)[1]
    end
        
    res = optimize(neg_lognormal_pdf, initial_guess, BFGS())
    est_mode = res.minimizer
    println("Task A.2.d.iv_____________________")
    println("estimated mode = " * string(est_mode))
    

end



if true # this is task B

    function read_csv(fname)  
        fin = open(fname,"r")
        lines = readlines(fin)
        close(fin)
    
        D = zeros(4,4) # create an 4x4 matrix with zeros
        for line in lines[2:end] # Loop through all lines. skip line 0 as it is just the header
            
            line_data = split(replace(line,raw"\r\n"=>""),",")
            hair = line_data[1] 
            eye = line_data[2]
            count = parse(Int,line_data[3])
    
    
            # A nicer alternative is to use a dictionary instead of the below if-statements
            # but a dictionary might be more difficult to interpret if you are new to julia
            if hair == "Black"
                c = 1 # use column 1
            end
            if hair == "Brown"
                c = 2 # use column 2 etc.
            end
            if hair == "Red"
                c = 3
            end
            if hair == "Blond"
                c = 4
            end
    
    
            if eye == "Brown"
                r = 1 # use row 1
            end
            if eye == "Blue"
                r = 2 # use row 2 etc.
            end
            if eye == "Hazel"
                r = 3
            end
            if eye == "Green"
                r = 4
            end
    
            # store count in matrix at row r and column c:
            D[r,c] = count
        end
        # normalize 
        D = D/float(sum(D))
        return D
    end

    fname = (@__DIR__)*"/data/HairEyeColor.csv"
    D = read_csv(fname)
    println("Task B.1.d________________________")        
    println(D)        
    println("Task B.2.a________________________")        
    println(sum(D,dims=2))
    println("Task B.2.b________________________")        
    println(sum(D,dims=1))
    println("Task B.2.c________________________")        
    println(sum(D))
        
    
    
    println("Task B.3.a________________________")
    println(D[2,4])
    
    println("Task B.3.b________________________")        
    println(sum(D[1,:]))
        
    println("Task B.3.c________________________")        
    println("p(red,brown)/p(brown)")    
    println(D[1,3]/sum(D[1,:]))
        
    
    println("Task B.3.d________________________")        
    println("Intercection of colums 3,4 and rows 1,2")
    println(sum(D[1:2,3:end]))
    
    println("Task B.3.e________________________")        
    println("union of colums 3,4 and rows 1,2")
    println("remeber P(AUB)=P(A)+P(B)-P(A&B)")
    println(sum(D[1:2,:]) + sum(D[:,3:end])-sum(D[1:2,3:end]))
    println("you can also take the complement")
    println(1 - sum(D[3:end,1:2]))
    

end
