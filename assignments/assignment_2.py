# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:03:43 2017

@author: jesper
"""


from __future__ import division

from pylab import * # this line will import math and plot functions that I may need

close('all')



if 1: # this is Task A:1

    def coin_tosses(theta,n):
        u = rand(n)
        c = zeros(n)
        c[u<theta] = 1.0
        return c

    n = 1000
    theta = 0.5
    c = coin_tosses(theta,n)    

    nr_heads = 0
    prop_heads = zeros(n)
    for i in range(n):
        nr_heads += c[i]
        prop_heads[i] = nr_heads/float(i+1)

    figure()
    semilogx(prop_heads,'bo-')
    plot(r_[0:n],theta*ones(n),'k:')
    title('Task A.1.a: recreating Figure 4.1')
    
    
    n = 100
    theta = 0.25
    c = coin_tosses(theta,n)     
    
    figure()
    hist(c,[-0.5,0.5,1.5],normed=True,label='Histogram from n=%d'%n+' samples')
    stem([0,1],[1-theta,theta],'k',label='True PMF')
    legend()
    title('Task A.1.c')
    





if 1:
    n = 10000
    mu = 3.4
    sigma = sqrt(3)
    x = mu+sigma*randn(n)

        
    
    def normal_pdf(x,mu,sigma): # define a normal probabolity distribution
        y = 1/sqrt(2*pi*sigma**2)*exp(-0.5*(x-mu)**2/sigma**2)
        return y
    
    bins = r_[mu-4*sigma:mu+4*sigma:0.1]
    figure()
    hist(x,bins,normed=True,label='Histogram from n=%d'%n+' samples')

    x_values = r_[mu-4*sigma:mu+4*sigma:0.1]
    plot(x_values,normal_pdf(x_values,mu,sigma),'k-',label=r'$\mathrm{N}(\mu=3.4,\sigma=\sqrt{3})$')
    leg = plt.legend(fancybox=True)
    leg.get_frame().set_alpha(0.5)
    title('Task A.2.a')    
    
    dx = 0.1
    x_values = r_[mu-6*sigma:mu+6*sigma:dx]
    print('Task A.2.b_________________________')
    print('sample mean = ' + str(mean(x)))
    print('Riemann sum and Eq. (4.6) = ' + str(np.sum(x_values*normal_pdf(x_values,mu,sigma)*dx)))
    print('True mu = ' + str(mu))
    
    print('Task A.2.c_________________________')
    print('sample variace = ' + str(var(x)))
    int_mu = np.sum(x_values*normal_pdf(x_values,mu,sigma)*dx)
    print('Riemann sum and Eq. (4.8) = ' + str(np.sum((x_values-int_mu)**2*normal_pdf(x_values,mu,sigma)*dx)))
    print('True mu = ' + str(sigma**2))
    
    

    n = 10000
    mu = 0
    sigma = 1
    x = mu+sigma*randn(n)
    y = exp(x)

    def lognormal_pdf(x,mu,sigma):
        y = 1.0/x/sqrt(2*pi*sigma**2)*exp(-0.5*(log(x)-mu)**2/sigma**2)
        return y    

    bins = r_[0:10:0.1]-0.05
    figure()
    hist(y,bins,normed=True,label='Histogram from n=%d'%n+' samples')

    x_values = r_[0:10:0.01]
    plot(x_values,lognormal_pdf(x_values,mu,sigma),'k-',label=r'$\mathrm{Lognormal}(\mu=0,\sigma=1)$')
    leg = plt.legend(fancybox=True)
    leg.get_frame().set_alpha(0.5)
    title('Task A.2.d.ii')        

    z = lognormal_pdf(y,mu,sigma)
    sample_mode = y[argmax(z)]
    print('Task A.2.d.iii_____________________')
    print('sample mode = ' + str(sample_mode))
    
    
    
    from scipy.optimize import fmin
    initial_guess = 2
    def neg_lognormal_pdf(x):
        return -lognormal_pdf(x,0,1)
        
    est_mode = fmin(neg_lognormal_pdf,initial_guess)
    print('Task A.2.d.iv_____________________')
    print('estimated mode = ' + str(est_mode))
    



if 1: # this is task B

    def read_csv(fname):  
        fin = open(fname,'r')
        lines = fin.readlines()
        fin.close()
    
        D = zeros((4,4)) # create an 4x4 matrix with zeros
        for line in lines[1:]: # Loop through all lines. skip line 0 as it is just the header
            
            line_data = line.replace('\r\n','').split(',')
            hair = line_data[0] 
            eye = line_data[1]
            count = int(line_data[2])
    
    
            # A nicer alternative is to use a dictionary instead of the below if-statements
            # but a dictionary might be more difficult to interpret if you are new to python
            if hair == 'Black':
                c = 0 # use column 0
            if hair == 'Brown':
                c = 1 # use column 1 etc.
            if hair == 'Red':
                c = 2
            if hair == 'Blond':
                c = 3
    
    
            if eye == 'Brown':
                r = 0 # use row 0
            if eye == 'Blue':
                r = 1 # use row 1 etc.
            if eye == 'Hazel':
                r = 2
            if eye == 'Green':
                r = 3
    
            # store count in matrix at row r and column c:
            D[r,c] = count
    
        # normalize 
        D = D/float(np.sum(D))
        return D

    fname = '../data/HairEyeColor.csv'
    D = read_csv(fname)
    print('Task B.1.d________________________')        
    print(D)        
    print('Task B.2.a________________________')        
    print(np.sum(D,axis=1))
    print('Task B.2.b________________________')        
    print(np.sum(D,axis=0))
    print('Task B.2.c________________________')        
    print(np.sum(D))
        
    
    
    print('Task B.3.a________________________')
    print(D[1,3])
    
    print('Task B.3.b________________________')        
    print(np.sum(D[1,:]))
        
    print('Task B.3.c________________________')        
    print('p(red,brown)/p(brown)')    
    print(D[0,2]/np.sum(D[0,:]))
        
    
    print('Task B.3.d________________________')        
    print('Intercection of colums 2,3 and rows 0,1')
    print(np.sum(D[0:2,2:]))
    
    print('Task B.3.e________________________')        
    print('union of colums 2,3 and rows 0,1')
    print('remeber P(AUB)=P(A)+P(B)-P(A&B)')
    print(np.sum(D[0:2,:])+np.sum(D[:,2:])-np.sum(D[0:2,2:]))
    print('you can also take the complement')
    print(1-np.sum(D[2:,:2]))
    


