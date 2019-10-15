import time
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import norm as normal
import matplotlib.pyplot as plt
import sys


def slice_sample(x0, w, log_pdf, N = 1000, m = 1e9, printing = True): 
        # A direct implementation from this paper
        # https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461

        D = len(x0)
        xs = np.zeros((N,D))
        lp = np.zeros(N)

        if len(w.shape)==2:
            w = np.diag(w)

        for i in range(N):
            l = 1*x0                
            r = 1*x0                
            x1 = 1*x0

            for d in np.random.permutation(D):            
                lu = np.log(np.random.rand())
                u1 = np.random.rand()
                v1 = np.random.rand()  
                
                if i == 0:
                    y = log_pdf(x0) + lu
                    evals = 1
                else:
                    y = log_pdf_x1 + lu

                l[d] = x0[d] - u1*w[d] 
                r[d] = l[d] + w[d]
                
                j = np.floor(m*v1)
                k = (m-1)-j        
                while y < log_pdf(l) and j>0:
                    l[d] -= w[d]
                    j -= 1

                while y < log_pdf(r) and k>0:
                    r[d] += w[d]
                    k -= 1
                    
                while True:
                    u2 = np.random.rand()  
                    x1[d] = l[d] + u2*(r[d]-l[d])

                    log_pdf_x1 = log_pdf(x1)
                    evals += 1                    
                    if y <= log_pdf_x1:
                        x0[d] = x1[d]
                        break
                    elif x1[d]<x0[d]:
                        l[d] = x1[d]
                    elif x1[d]>x0[d]:
                        r[d] = x1[d]
                    else: 
                        print y
                        print log_pdf_x1
                        print x1
                        print x0
                        raise RuntimeError('shrinkage error')
                        
                   
            xs[i] = x1
            lp[i] = log_pdf_x1
            if printing:
                if i % int(N/100.0) == 0: 
                    sys.stdout.write('%d '%((i*100)/N))

        return xs, lp                


def slice_sample_mp(x0,w,log_pdf,N=10000,Nb=-1,printing=True, nr_processes = 4):
    if nr_processes == 1:
        return slice_sample(x0,w,log_pdf,N=N,printing=printing)

    else:
        if len(x0.shape) == 1:
            x0s = [x0 for i in range(nr_processes)]

        if Nb == -1:
            Nb = int(round(N*0.1))

        f = lambda x0: slice_sample(x0,w,log_pdf,N=N,printing=printing)
        res = mypool(f,x0s)
        x,lp = list_to_arrays(res,Nb=Nb)
        return x,lp


# If you use numpy and scipy linalg shit, then
# run spyder with one thread if we use multiprocessing.
# $ OMP_NUM_THREADS=1 spyder
import multiprocessing
def mypool(f,arg_list,seed=None):
    def worker(f, arg, return_list):
        '''worker function'''
        return_list.append(f(arg))
        
    if seed == None:
       seed = np.random.randint(1000)


    manager = multiprocessing.Manager()
    return_list = manager.list()
    jobs = []
    for i in range(len(arg_list)):
        np.random.seed(i+seed)
        p = multiprocessing.Process(target=worker, args=(f,arg_list[i],return_list))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    return return_list




def list_to_arrays(res,Nb=0):
    x,lp = [],[]            
    for r in res:
        if x == []:
            x=r[0][Nb:]                    
            lp=r[1][Nb:]
        else:
            x = np.vstack((x,r[0][Nb:]))                    
            lp = np.vstack((lp,r[1][Nb:]))
    return x,lp                    



def hdi(theta_samp,alpha=0.05):
    # A direct implementation from Kruschke 2014.
    cred_mass = 1-alpha
    if np.rank(theta_samp)>1:
        N,K = theta_samp.shape    
        cis = np.zeros((K,2))    
        for k in range(K):
            
            ts = theta_samp[:,k]
            sind = np.argsort(ts)
            sts = ts[sind]
        
            N = len(sind)
            length_ci = np.inf
            for i in np.r_[0:int(np.floor(N*alpha))]:
                i2 = int(np.floor(N*cred_mass)+i)
                prop_ci = [sts[i],sts[i2]]
                length_prop_ci = prop_ci[1]-prop_ci[0]
                if length_prop_ci < length_ci:
                    ci = prop_ci
                    length_ci = ci[1]-ci[0]
            cis[k] = ci
        return cis
    else:
        N = len(theta_samp)
        
        ts = theta_samp
        sind = np.argsort(ts)
        sts = ts[sind]
    
        N = len(sind)
        length_ci = np.inf
        for i in np.r_[0:int(np.floor(N*alpha))]:
            i2 = int(np.floor(N*cred_mass)+i)
            prop_ci = [sts[i],sts[i2]]
            length_prop_ci = prop_ci[1]-prop_ci[0]
            if length_prop_ci < length_ci:
                ci = prop_ci
                length_ci = ci[1]-ci[0]
        return ci


    


if __name__ == '__main__':
    
    close('all')
    np.random.seed(0)

    def log_pdf(x):
        mu1 = array([1,2])
        mu2 = array([-1,2])
        C1 = array([[ 5,  5],[ 5, 10.0]])/10.0
        C2 = array([[ 10,  -5],[-5, 5.0]])/10.0
        x1 = x-mu1
        x2 = x-mu2
        return log(0.5*exp(-dot(dot(x1,inv(C1)),x1))+0.5*exp(-dot(dot(x2,inv(C2)),x2)))

    theta_start = ones(2)
    jump_step = ones(2)

    x,l = slice_sample_mp(theta_start,jump_step,log_pdf,N=10000, nr_processes=1) 

    figure()
    plot(x[:,0],x[:,1],'.')
    xlabel('$x_0$')
    ylabel('$x_1$')
    title('Scatter plot of the samples')
    axis([-3, 3, -1, 4])
    
    figure()
    plt.plot(x)
    xlabel('$i$')
    ylabel('$x[i]$')
    title('Trace plot of the samples')

     
                    
    
        