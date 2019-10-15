# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:03:43 2017

@author: jesper
"""


from __future__ import division
from pylab import * # this line will import math and plot functions that I may need
from pystat_mcmc import slice_sample_mp # to import slice_sample_mp (mp=multi processing version)

close('all')

y = array([1,0,1,1,0,1,1,1,0,1,1,1,1]) # coin flips 

if 1: # sample from posterior 
    
    def bernoulli_log_pdf(y,theta): # Eq. (6.1)
        return log(theta)*y + log(1.0-theta)*(1.0-y)
    
    def log_likelihood(y,theta): # Eq. (6.2) first row
        return np.sum(bernoulli_log_pdf(y,theta))

    def prior_log_pdf(theta): # It is handy to define valid regions using log and boolean expressions. 
        return log(0<theta<1) 
         
    def posterior_log_pdf(theta):
        return log_likelihood(y,theta) + prior_log_pdf(theta)
    
    theta_start = array([0.5]) # start value of theta, must be 0<theta<1   
    jump = array([0.1]) # jump length in slice sampling, must be less than 1 in this example otherwise it always jumps out of 0<theta<1  
    N = 1000 # number of samples
    nr_processes = 1 # number of chains to run in parallell
    theta_samples,posterior_log_pdf_values = slice_sample_mp(theta_start,jump,posterior_log_pdf,N=N,nr_processes=nr_processes)
    
    figure()
    subplot(121)
    hist(theta_samples,int(sqrt(N)),normed=True)
    title('Histogram of posterior samples')
    xlim([0,1])
    xlabel(r"$\theta$")    
    ylabel(r"$p(\theta|y)$")
    
    subplot(122)
    plot(theta_samples)
    xlabel(r'$i$ (sample index)')
    ylabel(r'$\theta^{\{i\}}$')
    title('Trace plot of posterior samples')
    

if 1:
    
    import pystan
    
    ass4_code = """
    data {
        int<lower=0> J; // number of flips
        int<lower=0,upper=1> y[J]; // coin flips
    }
    parameters {
        real<lower=0,upper=1> theta; // prob of getting a head 
    }
    transformed parameters {
    // no transformed variables to use
    }
    model {
        theta ~ beta(1, 1); // prior distribution for theta
        y ~ bernoulli(theta); // likelihood, note that stan will create the posterior automatically. 
    }
    """
    
    ass4_dat = {'J': len(y),
                   'y': y}

    
    fit = pystan.stan(model_code=ass4_code, data=ass4_dat,
                      iter=1000, chains=1)
    
    print(fit)
    
    fit.plot()
    
    la = fit.extract(permuted=False)  # return a dictionary of arrays
    
    
