
from math import exp, log
from scipy.stats import multivariate_normal as mvnorm
import numpy as np 



def resampling(w,N):
    # Algorithm 10: Resampling
    # output: index array of chosen particles 
    np.random.seed(123345)     
    I = [-1]*N #dummy array    
    w_hat = [0]
    w_hat[0] =  w[0]   
    for j in range(1,N):
        w_hat.append(sum(w[0:j+1]))        
    random_num = np.random.uniform(low=0,high=1/N,size=1)
    k = 0  
    for j in range (N):
        while random_num > w_hat[k]:
            k += 1
        I[j] = k
        random_num =  random_num +1/N
        k = 0       
    return I


def gaussian_log_likelihood(data, trial, measurement_noise):
    arg = data - trial
    f = mvnorm.pdf(arg, mean=None, cov=measurement_noise)
    return log(f)


def particle_std_start(num_particles, prior, measurement_model, log_likelihood, data, measurement_noise):
    ## num_particles is the number of testing points we want to use
    ## prior is a function which samples from the prior distribution
    ## measurement_model is the "modeler's model", not the ground-truth dynamics
    ## log_likelihood is what it says it is; for now we're taking likelihood as Gaussian
    ##      the variance of this Gaussian is the measurement_noise argument, and
    ##      comes from the ground-truth model
    ## data also comes from the ground-truth model
    points = []
    for j in range(num_particles):
        points.append(prior())  # generate particles

    weights = []
    for j in range(num_particles):
        trial = measurement_model(points[j])
        weights[j] = log_likelihood(data, trial, measurement_noise)

    # normalize
    w_max = max(weights)
    for j in range(num_particles):
        weights[j] = exp(weights[j] - w_max)

    w_sum = sum(weights)
    for j in range(weights):
        weights[j] = weights[j] / w_sum

    return points, weights


def particle_std_iter(num_particles, points, measurement_model, log_likelihood, data, measurement_noise):
    ## same as the previous function, but now points is an argument that comes from a previous
    ##      step of the algorithm

    weights = []
    for j in range(num_particles):
        trial = measurement_model(points[j])
        weights[j] = log_likelihood(data, trial, measurement_noise)

    # normalize
    w_max = max(weights)
    for j in range(num_particles):
        weights[j] = exp(weights[j] - w_max)

    w_sum = sum(weights)
    for j in range(weights):
        weights[j] = weights[j] / w_sum

    return points, weights


def particle_std(num_particles, num_iter, prior, measurement_model, log_likelihood, data, measurement_noise):
    estimates = []
    points, weights = particle_std_start(num_particles, prior, measurement_model,
                                         log_likelihood, data, measurement_noise)
    estimates.append(points)
    for k in range(num_iter):
        I = resampling(weights,num_particles)
        ## assign new particles according to I, which is a list of indices
        new_points = []
        for j in range(num_particles):
            index = I[j]
            new_points.append(points[index])
        ## run the algorithm again
        points, weights = particle_std_iter(num_particles, new_points, measurement_model,
                                            log_likelihood, data, measurement_noise)
        estimates.append(points)

    return estimates
