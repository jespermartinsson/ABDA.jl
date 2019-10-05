# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:03:43 2017

@author: jesper
"""


from __future__ import division
from pylab import * # this line will import math and plot functions that I may need


close('all')




if 0: # this is task A.1

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
    print(D)        
        
    print('Task B.3.c________________________')        
    print('p(red,blue)/p(blue)')    
    print(D[1,2]/np.sum(D[1,:]))
        
    
if 0: # this is task A.2
    n = 10000
    P = {}

    P["large"] = 5.0/n
    P["small"] = 1.0-P["large"]
    
    P["alarm|large"] = 99/100.0    
    P["alarm|small"] = 1-98/100.0

    P["alarm"] = P["alarm|large"]*P["large"] + P["alarm|small"]*P["small"]
    
    P["large|alarm"] = P["alarm|large"]*P["large"] / P["alarm"]
    print('Task A.2.d________________________')        
    print('P["large|alarm"]')  
    print  P["large|alarm"]

        
    print('\nP["small|alarm"]')  
    P["small|alarm"] = P["alarm|small"]*P["small"] / P["alarm"]
    print  P["small|alarm"]
    
    
    print '\nWe annually expect 10000*P["small"] events to be small, then' 
    print 'we expect P["alarm|small"] * 10000*P["small"] ='
    print  P["alarm|small"] * 10000 *P["small"]
    print 'false alarms' 
    
    def cost(X):
        return P["alarm|small"] *10000*P["small"]*X
        
    X = r_[100:1001:1]
    figure()
    plot(X,cost(X))    
    xlabel('Evacuation cost $X$ kkr')
    ylabel('False alarm cost kkr')
    


if 1: # this is task A.2
    P = {}

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
    


    def post1(T): # version 1 just to update likelihood keeping the same prior 
        prior = array([P[":("], P[":)"]])
        likelihood = array([1.0,1.0])
        for t in T:
            likelihood[0] = likelihood[0]*P[t+"|:("]
            likelihood[1] = likelihood[1]*P[t+"|:)"]
        
        posterior = likelihood*prior/np.sum(likelihood*prior)
        return posterior

    def post2(T): # version 2 is doing sequential Bayes updating the prior using the posterior of the previous test result
        # initialize prior and likelihood and posterior
        prior = array([P[":("], P[":)"]])
        likelihood = array([1.0,1.0])
        posterior = likelihood*prior/np.sum(likelihood*prior) # Bayes formula with no data posterior = prior

        for t in T: # if data in T we will enter the loop
            likelihood[0] = P[t+"|:("] # get likelihood for result in t given sick
            likelihood[1] = P[t+"|:)"] # get likelihood for result in t given healthy
            posterior = likelihood*prior/np.sum(likelihood*prior) # calc posterior
            prior = posterior # update prior until the next experiment
        return posterior


    T = '+-+'
    print(post1(T))
    print(post2(T))
    
    raise IOError


if 1:
    
    def bernoulli_pdf(y,theta): # Eq. (6.1)
        return theta**y*(1.0-theta)**(1.0-y)
    
    def likelihood(y,theta): # Eq. (6.2) first row
        return prod(bernoulli_pdf(y,theta))
    
    def bernoulli_log_pdf(y,theta): # Eq. (6.1)
        return log(theta)*y + log(1.0-theta)*(1.0-y)
    
    def log_likelihood(y,theta): # Eq. (6.2) first row
        return np.sum(bernoulli_log_pdf(y,theta))

    for n in [10,1000,100000]:
        y = np.round(rand(n))
        print likelihood(y,0.5), log_likelihood(y,0.5)
    
    for y in [array([1]),array([1,1]),array([1,1,0,1])]: 
        thetas = r_[0:1:0.01]
        likes = zeros(thetas.shape[0])
        for i in range(thetas.shape[0]):
            theta = thetas[i]
            likes[i] = likelihood(y,theta)
    
        figure()
        plot(thetas,likes)        
        
    
