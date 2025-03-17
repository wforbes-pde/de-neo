import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import logging
from numpy import array
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.linalg import svd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler  
import itertools

np.random.seed(42)
print_master = True

# https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube
def lhs_init(samples_, dim, bounds):

    # appears to be a samples of R^b
    # is this okay? 

    sample = lhs(n=dim, samples=samples_)
    pop = np.zeros((samples_, dim))
    for i in range(dim):
        pop[:, i] = sample[:, i] * bounds
        #plt.plot(np.arange(0,len(pop[0,:])),pop[0,:], linewidth=0.75)
        #plt.show()
        #a=False
    return pop


def mutate(d, NP, NP_indices, F_array, x):

    # random mutation with distinct indices
    
    # create base vector array

    base_array = np.full((d, NP), 1.0)
    v2_array = np.full((d, NP), 1.0)
    v1_array = np.full((d, NP), 1.0)

    # random mutation with distinct indices

    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 3, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        base = x[:,i].copy()
        base_array[:,e] = base
        v1 = x[:,j].copy()
        v1_array[:,e] = v1
        v2 = x[:,k].copy()
        v2_array[:,e] = v2

    p = base_array + F_array*(v2_array-v1_array)
    return p

def mutate_two(d,NP, NP_indices, F_array, F2_array, x):

    # random mutation with distinct indices

    # create base vector array

    base_array = np.full((d, NP), 1.0)
    v2_array = np.full((d, NP), 1.0)
    v1_array = np.full((d, NP), 1.0)
    v3_array = np.full((d, NP), 1.0)
    v4_array = np.full((d, NP), 1.0)
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 5, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        l = test[e][3]
        m = test[e][4]
        base = x[:,i].copy()
        base_array[:,e] = base
        v1 = x[:,j].copy()
        v1_array[:,e] = v1
        v2 = x[:,k].copy()
        v2_array[:,e] = v2
        v3 = x[:,l].copy()
        v3_array[:,e] = v3
        v4 = x[:,m].copy()
        v4_array[:,e] = v4           
        
    p = base_array + F_array*(v2_array-v1_array) + F2_array*(v4_array-v3_array)
    return p

def mutate_three(d, NP, NP_indices, F_array, F2_array, F3_array, x):

    # random mutation with distinct indices

    # create base vector array

    base_array = np.full((d, NP), 1.0)
    v2_array = np.full((d, NP), 1.0)
    v1_array = np.full((d, NP), 1.0)
    v3_array = np.full((d, NP), 1.0)
    v4_array = np.full((d, NP), 1.0)
    v5_array = np.full((d, NP), 1.0)
    v6_array = np.full((d, NP), 1.0)
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 5, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        l = test[e][3]
        m = test[e][4]
        n = test[e][4]
        o = test[e][5]
        base = x[:,i].copy()
        base_array[:,e] = base
        v1 = x[:,j].copy()
        v1_array[:,e] = v1
        v2 = x[:,k].copy()
        v2_array[:,e] = v2
        v3 = x[:,l].copy()
        v3_array[:,e] = v3
        v4 = x[:,m].copy()
        v4_array[:,e] = v4        
        v5 = x[:,n].copy()
        v5_array[:,e] = v5
        v6 = x[:,o].copy()
        v6_array[:,e] = v6        

    p = base_array + F_array*(v2_array-v1_array) + F2_array*(v4_array-v3_array) + F3_array*(v5_array-v6_array)
    return p


def mutate_best(d, NP, NP_indices, F_array, gen_best, x):

    # best mutation with distinct indices for each index

    # create base vector array

    base_array = np.full((d, NP), 1.0)
    v2_array = np.full((d, NP), 1.0)
    v1_array = np.full((d, NP), 1.0)

    # random mutation with distinct indices

    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 3, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        base_array[:,e] = gen_best
        v1 = x[:,j].copy()
        v1_array[:,e] = v1
        v2 = x[:,k].copy()
        v2_array[:,e] = v2

    p = base_array + F_array*(v2_array-v1_array)
    return p

def mutate_best_two(d, NP, NP_indices, F_array, F2_array, gen_best, x):

    # best mutation with distinct indices for each index
    
    # create base vector array

    base_array = np.full((d, NP), 1.0)
    v2_array = np.full((d, NP), 1.0)
    v1_array = np.full((d, NP), 1.0)
    v3_array = np.full((d, NP), 1.0)
    v4_array = np.full((d, NP), 1.0)
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 5, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        l = test[e][3]
        m = test[e][4]
        base = x[:,i].copy()
        base_array[:,e] = base
        v1 = x[:,j].copy()
        v1_array[:,e] = v1
        v2 = x[:,k].copy()
        v2_array[:,e] = v2
        v3 = x[:,l].copy()
        v3_array[:,e] = v3
        v4 = x[:,m].copy()
        v4_array[:,e] = v4           
        
    p = base_array + F_array*(v2_array-v1_array) + F2_array*(v4_array-v3_array)
    return p

def mutate_best_three(d, NP, NP_indices, F_array, F2_array, F3_array, gen_best, x):

    # best mutation with distinct indices
    
    # create base vector array

    base_array = np.full((d, NP), 1.0)
    v2_array = np.full((d, NP), 1.0)
    v1_array = np.full((d, NP), 1.0)
    v3_array = np.full((d, NP), 1.0)
    v4_array = np.full((d, NP), 1.0)
    v5_array = np.full((d, NP), 1.0)
    v6_array = np.full((d, NP), 1.0)
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 5, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        l = test[e][3]
        m = test[e][4]
        n = test[e][4]
        o = test[e][5]
        base_array[:,e] = gen_best
        v1 = x[:,j].copy()
        v1_array[:,e] = v1
        v2 = x[:,k].copy()
        v2_array[:,e] = v2
        v3 = x[:,l].copy()
        v3_array[:,e] = v3
        v4 = x[:,m].copy()
        v4_array[:,e] = v4        
        v5 = x[:,n].copy()
        v5_array[:,e] = v5
        v6 = x[:,o].copy()
        v6_array[:,e] = v6        

    p = base_array + F_array*(v2_array-v1_array) + F2_array*(v4_array-v3_array) + F3_array*(v5_array-v6_array)
    return p


def selection_vector(NP_indices, DE_model, X_train, y_train, gen, mindex,
              reg_flag, error_metric_, m, n1, n2, n3,
              x_points, z_points, MCMC, chain_vector, NN_model):
    
    # determine survival of target or trial vector
    # into the next generation
    i_accept = 0
    n_ = len(y_train)

    for j in NP_indices:

        xcandidates = split_candidate(x_points[j], m, n1, n2, n3)
        xW0, xW1, xW2, xW3, xB0, xB1, xB2, xB3 = xcandidates

        zcandidates = split_candidate(z_points[j], m, n1, n2, n3)
        zW0, zW1, zW2, zW3, zB0, zB1, zB2, zB3 = zcandidates
        
        zfit, zyb = DE_model.fitness(X_train, zW0, zW1, zW2, zW3, zB0, zB1, zB2, zB3, y_train, n_, error_metric_, reg_flag, NN_model)
        xfit, xyb = DE_model.fitness(X_train, xW0, xW1, xW2, xW3, xB0, xB1, xB2, xB3, y_train, n_, error_metric_, reg_flag, NN_model)

        if zfit <= xfit:
            x_points[j] = z_points[j].copy()
            i_accept = i_accept + 1

        # MCMC acceptance

        # likelihood ratio
        # uniform random number alpha

        run_mcmc = MCMC.run_mcmc
        burn_in = MCMC.burn_in
        ratio = np.minimum(1,xfit/zfit)
        alpha = random.uniform(0,1)

        # serial chain

        if run_mcmc and not MCMC.parallel_chain and j == mindex and gen > burn_in:
            chain_vector = MCMC.serial_chain_MCMC(gen, xfit, zfit, mindex, run_mcmc, ratio, burn_in, j,
                            MCMC, alpha, x_points, z_points, chain_vector)
            
        # if run_mcmc and MCMC.parallel_chain and gen > burn_in:
            
        #     # for each index chain

        #     W0[j], W1[j], W2[j], W3[j], b0[j], b1[j], b2[j], b3[j] = MCMC.parallel_chain_MCMC(gen, xfit, zfit, mindex, run_mcmc, ratio, burn_in,j,
        #                     MCMC, alpha, x_points, z_points, 
        #                     W0[j], W1[j], W2[j], W3[j], b0[j], b1[j], b2[j], b3[j])

    return x_points, i_accept, chain_vector

def crossover_vector(NP_indices, y, x, CR):

    # at least one column vector swapped based on random int k

    z = x.copy()    

    x_ = x.copy()
    y_ = y.copy()
    z_ = z.copy()

    m,n = x_.shape

    for i in np.arange(0,m):
        k = np.random.choice(np.arange(0,n),) # think this should be n
        for j in np.arange(0,n):
            if (random.uniform(0, 1) <= CR[i,j] or j == k): # think this should be j
                z_[i,j] = y_[i,j].copy()
            else:
                z_[i,j] = x_[i,j].copy()

    return z_


def create_crossover_vector(x_, m, n1, n2, n3, CR_W0, CR_W1, CR_W2, CR_W3, CR_b0, CR_b1, CR_b2, CR_b3):

    CR = np.ones(len(x_))

    # F_1 mutation vector

    w0 = m*n1
    CR[0:w0] = CR_W0

    w1 = w0 + n1*n2
    CR[w0:w1] = CR_W1

    w2 = w1 + n2*n3
    CR[w1:w2] = CR_W2

    w3 = w2 + n3
    CR[w2:w3] = CR_W3

    b0 = w3 + n1
    CR[w3:b0] = CR_b0

    b1 = b0 + n2
    CR[b0:b1] = CR_b1

    b2 = b1 + n3
    CR[b1:b2] = CR_b2

    b3 = b2 + 1
    CR[b2:b3] = CR_b3
    
    return CR

def mutation_vector(d, NP, NP_indices, F_1, F_2, F_3, x_weight, gen_best_x_weight, mutation_type):

    if mutation_type == 'random':

        y = mutate(d, NP, NP_indices, F_1, x_weight)

    # DE/rand/2 needs minimum NP = 6

    if mutation_type == 'random2':

        y = mutate_two(d, NP, NP_indices, F_1, F_2, x_weight)

    # DE/rand/3 needs minimum NP = 8

    if mutation_type == 'random3':

        y = mutate_three(d, NP, NP_indices, F_1, F_2, F_3, x_weight)
    
    # DE/best/123

    if mutation_type in ['best']:

        y = mutate_best(d, NP, NP_indices, F_1, gen_best_x_weight, x_weight)

    if mutation_type in ['best2']:

        y = mutate_best_two(d, NP, NP_indices, F_1, F_2, gen_best_x_weight, x_weight)

    if mutation_type in ['best3']:

        y = mutate_best_three(d, NP, NP_indices, F_1, F_2, F_3, gen_best_x_weight, x_weight)

    return y


def create_mutation_vector(x_, m, n1, n2, n3, F_one, F_two, F_three):

    F_W0, F_W1, F_W2, F_W3, F_b0, F_b1, F_b2, F_b3 = F_one
    F2_W0, F2_W1, F2_W2, F2_W3, F2_b0, F2_b1, F2_b2, F2_b3 = F_two
    F3_W0, F3_W1, F3_W2, F3_W3, F3_b0, F3_b1, F3_b2, F3_b3 = F_three

    F_1 = np.ones(len(x_))
    F_2 = np.ones(len(x_))
    F_3 = np.ones(len(x_))

    # F_1 mutation vector

    w0 = m*n1
    F_1[0:w0] = F_W0

    w1 = w0 + n1*n2
    F_1[w0:w1] = F_W1

    w2 = w1 + n2*n3
    F_1[w1:w2] = F_W2

    w3 = w2 + n3
    F_1[w2:w3] = F_W3

    b0 = w3 + n1
    F_1[w3:b0] = F_b0

    b1 = b0 + n2
    F_1[b0:b1] = F_b1

    b2 = b1 + n3
    F_1[b1:b2] = F_b2

    b3 = b2 + 1
    F_1[b2:b3] = F_b3

    # F_2 mutation vector

    F_2[0:w0] = F2_W0
    F_2[w0:w1] = F2_W1
    F_2[w1:w2] = F2_W2
    F_2[w2:w3] = F2_W3
    F_2[w3:b0] = F2_b0
    F_2[b0:b1] = F2_b1
    F_2[b1:b2] = F2_b2
    F_2[b2:b3] = F2_b3

    # F_3 mutation vector

    F_3[0:w0] = F3_W0
    F_3[w0:w1] = F3_W1
    F_3[w1:w2] = F3_W2
    F_3[w2:w3] = F3_W3
    F_3[w3:b0] = F3_b0
    F_3[b0:b1] = F3_b1
    F_3[b1:b2] = F3_b2
    F_3[b2:b3] = F3_b3
    
    return F_1, F_2, F_3


def generate_initial_population(d, NP, NP_indices, init):
    
    # bias matrix initialization based on 
    # a number of rows, n_input
    # b number of columns, n_output
    # NP number of candidates
    
    # init values
    
    itype,l,h = init

    # Kaiming He weight initialization for relu
    # loc = mean, scale = standard deviation    

    if itype == 'uniform':
        x =np.random.uniform(low=l, high=h, size=(d, NP))

    # candidates = {}

    # if itype == 'latin':
    #     bounds=1
    #     x = lhs_init(len(NP_indices), a*b, bounds)
    #     for j in NP_indices:
    #         trans = x[j,:].reshape(len(x[j,:]),1)
    #         xx = trans.reshape(a,b)
    #         candidates[j] = xx

    # if itype == 'halton':
    #     sampler = qmc.Halton(d=a*b, scramble=True)
    #     x = sampler.random(n=len(NP_indices))
    #     for j in NP_indices:
    #         trans = x[j,:].reshape(len(x[j,:]),1)
    #         xx = trans.reshape(a,b)
    #         candidates[j] = xx

    #     if itype == 'he':
    #         mean_ = 0
    #         sigma = np.sqrt(2.0) * np.sqrt(2 / (a+b))
    #         x = np.random.normal(loc=mean_, scale=sigma, size=(a,b))
    #     candidates[j] = x
    
    return x

# X_train, y_train,
def selection(NP_indices, x_dict, y_dict, gen, mindex, 
              current, reg_flag, error_metric_, 
              W0, W1, W2, W3, b0, b1, b2, b3,
              x_points, z_points, bootstrapping, MCMC, NN_model, DE_model):
    
    x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3 = x_points
    z_W0, z_W1, z_W2, z_W3, z_b0, z_b1, z_b2, z_b3 = z_points
    
    # determine survival of target or trial vector 
    # into the next generation
    
    if not bootstrapping:
        n_ = len(y_dict)
        X_train = x_dict
        y_train = y_dict
        
    i_accept = 0

    for j in NP_indices:
        if bootstrapping:
            X_train = x_dict[j]
            y_train = y_dict[j]
            n_ = len(y_train)
        
        zfit, zyb = DE_model.fitness(X_train, z_W0[j], z_W1[j], z_W2[j], z_W3[j], z_b0[j], z_b1[j], z_b2[j], z_b3[j], y_train, n_, error_metric_, reg_flag, NN_model)
        xfit, xyb = DE_model.fitness(X_train, x_W0[j], x_W1[j], x_W2[j], x_W3[j], x_b0[j], x_b1[j], x_b2[j], x_b3[j], y_train, n_, error_metric_, reg_flag, NN_model)

        if zfit <= xfit:
            
            x_W0[j] = z_W0[j].copy()
            x_W1[j] = z_W1[j].copy()
            x_W2[j] = z_W2[j].copy()
            x_W3[j] = z_W3[j].copy()
            x_b0[j] = z_b0[j].copy()
            x_b1[j] = z_b1[j].copy()
            x_b2[j] = z_b2[j].copy()
            x_b3[j] = z_b3[j].copy()
            i_accept = i_accept + 1 
        
        # MCMC acceptance

        # likelihood ratio
        # uniform random number alpha

        run_mcmc = MCMC.run_mcmc
        burn_in = MCMC.burn_in
        ratio = np.minimum(1,xfit/zfit)
        alpha = random.uniform(0,1)

        # serial chain

        if run_mcmc and not MCMC.parallel_chain and j == mindex and gen > burn_in:
            W0, W1, W2, W3, b0, b1, b2, b3 = MCMC.serial_chain_MCMC(gen, xfit, zfit, mindex, run_mcmc, ratio, burn_in, j,
                            MCMC, alpha, x_points, z_points, 
                            W0, W1, W2, W3, b0, b1, b2, b3)
            
        if run_mcmc and MCMC.parallel_chain and gen > burn_in:
            
            # for each index chain

            W0[j], W1[j], W2[j], W3[j], b0[j], b1[j], b2[j], b3[j] = MCMC.parallel_chain_MCMC(gen, xfit, zfit, mindex, run_mcmc, ratio, burn_in,j,
                            MCMC, alpha, x_points, z_points, 
                            W0[j], W1[j], W2[j], W3[j], b0[j], b1[j], b2[j], b3[j])
    
    selected_points = x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3
    return selected_points, i_accept, W0, W1, W2, W3, b0, b1, b2, b3

def differential_evolution_vector(DE_model, train_size_):
    
    NP = DE_model.NP
    G = DE_model.g
    F = DE_model.F
    CR = DE_model.CR
    mutation_type = DE_model.mutation_type
    return_method = DE_model.return_method    
    error_metric = DE_model.error_metric
    run_enh = DE_model.run_enh
    run = DE_model.run
    d = DE_model.d
    bootstrapping, ratio_ = DE_model.bootstrapping

    # parameters

    F_delta = DE_model.F_delta
    tol = DE_model.tol
    NPI = np.maximum(DE_model.NPI,DE_model.NP)
    init = DE_model.init
    lowerF = DE_model.lowerF
    upperF = DE_model.upperF
    track_len = DE_model.track_length
    refine_gen_start, refine_current_start, refine_mod_start, refine_random = DE_model.refine_param
    F_refine = DE_model.F_refine    
    mutation_refine = DE_model.mutation_refine
    run_exh, exh_current_start, exh_subset = DE_model.exhaustive

    CR_refine = DE_model.CR_refine
    lowerCR = DE_model.lowerCR
    upperCR = DE_model.upperCR
    CR_delta = DE_model.CR_delta   

    # enhancement
    
    #one, two = reg_flag
    run_svd, run_cluster, run_local = run_enh  
    
    NP_indices = list(np.arange(0,NP))
    initial_NP_indices = list(np.arange(0,NPI))
    df_list=[]
    accept_list = []

    # training data tracking for refinement

    gen_train_fitness_list = []
    gen_train_resid_list = []

    # validation data tracking for exit

    gen_val_fitness_list = []
    gen_val_resid_list = []

    gen_val_score = 0
    val_residual = 0
    val_run_avg_resid = 0
    vimin_value = 1e5
    val_sample = DE_model.val_sample

    d_gen_val_fitness_list, d_val_residual, d_gen_val_resid_list, d_val_run_avg_resid = {},{},{},{}
    for k in NP_indices:
        d_gen_val_fitness_list[k], d_val_residual[k], d_gen_val_resid_list[k], d_val_run_avg_resid[k] = [],{},[],{}
    
    # start DE exploration
    global acceptance_rate
    global a_rate

    skip = refine_mod_start-1
    svd_filter_r = [skip]
    svd_scalar_r = [skip]
    svd_exp_r = [skip]
    cluster_r =[skip]
    local_r = [skip]

    # dimensions
    
    for i in np.arange(0,G):

        # generate initial population for each weight matrix

        if i == 0:
            logging.info(f'run {run} gen {i} initial population start {DE_model.init[0]}')
            
            if init[0] in ['he', 'uniform', 'halton', 'latin']:

                ix_ = generate_initial_population(d, NP, initial_NP_indices, init)
            
            # initial population fitness

            initial_fitness = DE_model.analytical(ix_,d)

            # find best initial generation candidates

            iidx = np.argpartition(initial_fitness, NP-1)[:NP]

            imin_value = np.amin(initial_fitness)
            imindex = np.where(initial_fitness == imin_value)
            imindex = imindex[0][0]
            mindex = imindex

            # populate initial generation with best candidates

            x = np.full((d, NP), 1.0)

            for k in NP_indices:
                x[:,k] = ix_[:,iidx[k]]

            # set gen best

            gen_best_x = ix_[:,imindex]

            initial_fitness.sort()
            w = np.mean(initial_fitness[:NP])
            logging.info(f'gen {i} best fitness is {initial_fitness[0]}, avg fitness {w}, NPI {NPI}')

            min_value = 500
            
        if i > 0:

            x = xgp.copy()
            gen_best_x = gb_x.copy()

        if i < track_len:
            train_run_avg_residual_rmse = -1

        if train_run_avg_residual_rmse < tol:
            current = 0

        # mutation parameters

        F = DE_model.F
        CR = DE_model.CR
        mutation_type = DE_model.mutation_type
        tol = DE_model.tol
        lowerF = DE_model.lowerF
        upperF = DE_model.upperF
        F_delta = DE_model.F_delta
        error_metric = DE_model.error_metric

        # default CR

        CR_array = DE_model.return_F_CR('default', lowerF, upperF, CR_delta, DE_model.CR, d, NP)

        # default F values

        F_array = DE_model.return_F_CR('default', lowerF, upperF, CR_delta, DE_model.F, d, NP)
        F2_array = DE_model.return_F_CR('default', lowerF, upperF, CR_delta, DE_model.F, d, NP)
        F3_array = DE_model.return_F_CR('default', lowerF, upperF, CR_delta, DE_model.F, d, NP)

        # default mutation type
        
        mutation_list = DE_model.return_mutation_list(NP)    
        mutation_op = DE_model.return_mutation_type('default', mutation_list, DE_model.mutation_type)

        # refinement steps

        if train_run_avg_residual_rmse >= tol and i >= track_len-1:

            current = current + 1
            
            # randomly selection F, mutation, and crossover variation scheme under refinement
            
            if current > refine_current_start and i > refine_gen_start and refine_random:
                
                variation_list = ['default', 'variable', 'weight_variable']
                F_refine = random.choice(variation_list)
                CR_refine = random.choice(variation_list)
                mutation_refine = random.choice(variation_list)

            if F_refine == 'variable' and current > refine_current_start and i > refine_gen_start:

                F_array = DE_model.return_F_CR('variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)
                F2_array = DE_model.return_F_CR('variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)
                F3_array = DE_model.return_F_CR('variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)

            if CR_refine == 'variable' and current > refine_current_start and i > refine_gen_start:

                CR_array = DE_model.return_F_CR('variable', lowerF, upperF, CR_delta, DE_model.CR, d, NP)

            if mutation_refine == 'variable' and current > refine_current_start and i > refine_gen_start:
                
                mutation_op = DE_model.return_mutation_type('variable', mutation_list, DE_model.mutation_type)
        
        # mutation

        y = mutation_vector(d, NP, NP_indices, F_array, F2_array, F3_array, x, gen_best_x, mutation_op)
                
        # crossover

        z = crossover_vector(NP_indices, y, x, CR_array)
        
        # selection

        xgp, i_accept, chain_vector = selection_vector(NP_indices, DE_model, X_train, y_train,
                                                   i, mindex, reg_flag, DE_model.error_metric,
                                                   m, n1, n2, n3,
                                                   x, z, MCMC, chain_vector, NN_model)
        
        # if not bootstrapping:
        #     selected_points, i_accept, W0, W1, W2, W3, b0, b1, b2, b3 = selection(NP_indices, X_train, y_train,
        #                                                 i, mindex, current, reg_flag, DE_model.error_metric, 
        #                                                 W0, W1, W2, W3, b0, b1, b2, b3,
        #                                                 x_points, z_points, bootstrapping, MCMC, NN_model, DE_model)
        
        # if bootstrapping:
        #     selected_points, i_accept, W0, W1, W2, W3, b0, b1, b2, b3 = selection(NP_indices, x_dict, y_dict,
        #                                                i, mindex, current, reg_flag, DE_model.error_metric, 
        #                                                W0, W1, W2, W3, b0, b1, b2, b3,
        #                                                x_points, z_points, bootstrapping, MCMC, NN_model, DE_model)
        
        #xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = selected_points
        
        accept_list.append(i_accept)

        # fitness evaluation

        errors = []

        n_ = len(y_train)
        for j in NP_indices:
            candidates = split_candidate(x[j], m, n1, n2, n3)
            W0, W1, W2, W3, b0, b1, b2, b3 = candidates

            init_rmse, iyb = DE_model.fitness(X_train, W0, W1, W2, W3, b0, b1, b2, b3,
                                y_train, n_, error_metric, reg_flag, NN_model)

            errors.append(init_rmse)
        
        # for j in NP_indices:         
        #     if bootstrapping:
        #         X_train = x_dict[j]
        #         y_train = y_dict[j]
        #         m_ = len(y_train)
        #         gen_train_score, yb = DE_model.fitness(x_dict[j], xgp_W0[j], xgp_W1[j], xgp_W2[j], xgp_W3[j], xgp_b0[j], xgp_b1[j], xgp_b2[j], xgp_b3[j], 
        #                         y_dict[j], m_, DE_model.error_metric, reg_flag, NN_model)
        #     else:
        #         gen_train_score, yb = DE_model.fitness(X_train, xgp_W0[j], xgp_W1[j], xgp_W2[j], xgp_W3[j], xgp_b0[j], xgp_b1[j], xgp_b2[j], xgp_b3[j], 
        #                     y_train, n_, DE_model.error_metric, reg_flag, NN_model)
        #     errors.append(gen_train_score)

        # determine best generation point

        gen_fitness_values = np.array(errors)
        min_value = np.amin(gen_fitness_values)
        mindex = np.where(gen_fitness_values == min_value)
        mindex = mindex[0][0] # index integer

        # determine worst generation point

        max_value = np.amax(gen_fitness_values)
        maindex = np.where(gen_fitness_values == max_value )
        maindex = maindex[0][0]

        # define generation best

        gb_x = xgp[mindex]

        # training residual tracking

        gen_train_fitness_list, train_residual, gen_train_resid_list, train_run_avg_residual_rmse = \
            DE_model.return_running_avg_residual(i, min_value, gen_train_fitness_list, gen_train_resid_list, track_len)
        
        if train_residual > 0:
            logging.info(f'run {run} gen {i} index {mindex} {error_metric} {min_value} train resid {train_residual} val resid {val_residual} current {current}')
            #breakpoint()

        # refinement

        c_min_value = 0
        l_fit = 0
        
        svd_fit = 0
        s_scalar_value = 0
        s_exp_value = 0

        # SVD filter

        comparison_value = min_value

        if current > refine_current_start and i_accept > 0 and run_svd and current % refine_mod_start in svd_filter_r:
            gen_points = xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3
            svd_points, svd_fit  = DE_model.perform_svd_filter(NP,bootstrapping, x_dict, y_dict, comparison_value, maindex, DE_model,
                            reg_flag, NN_model, n_, gen_points,i, NP_indices, current)
            xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = svd_points
        
        # scalar SVD

        if current > refine_current_start and i_accept > 0 and run_svd and current % refine_mod_start in svd_scalar_r:
            gen_points = xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3
            svd_scalar_points, s_scalar_value  = DE_model.perform_svd_filter(NP,bootstrapping, x_dict, y_dict, comparison_value, maindex, DE_model,
                                    reg_flag, NN_model, n_, gen_points,i, NP_indices, current)
            xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = svd_scalar_points
        
        # exp scalar

        if current > refine_current_start and i_accept > 0 and run_svd and current % refine_mod_start in svd_exp_r:
            gen_points = xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3
            svd_exp_points, s_exp_value  = DE_model.perform_svd_exp(NP,bootstrapping, x_dict, y_dict, comparison_value, maindex, DE_model,
                                            reg_flag, NN_model, n_, gen_points,i, NP_indices, current)
            xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = svd_exp_points
        
        # clustering

        if current > refine_current_start and i_accept > 0 and run_cluster and current % refine_mod_start in cluster_r:
            cluster_points, c_min_value = DE_model.perform_clustering(NP,bootstrapping, x_dict, y_dict, comparison_value, maindex, DE_model,
                        reg_flag, NN_model, n_, xgp,i)            
            xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = cluster_points

        # local search
        # number of samples perturbed around population
        # fitness value of all samples checked.
        # perturbed candidate has fitness value lower than current poplation best, then it
        # replaces the minimum and process exit

        if current > refine_current_start and i_accept > 0 and run_local and current % refine_mod_start in local_r:
            search_points, l_fit = DE_model.perform_search(NP,bootstrapping, x_dict, y_dict,comparison_value, maindex, DE_model,
                        reg_flag, NN_model, n_, xgp,i, NP_indices, current)
            xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = search_points
        
        # collect parameters and data
        
        if refine_random:
            F_refine = 'random'
            CR_refine = 'random'
            mutation_refine = 'random'
        
        df = pd.DataFrame({'Run':[run], 'Generation':[i], 'F':[F], 'CR':[CR[0]], 'G':[G], 'NP':[NP], 'NPI':[NPI],'mutation_type':[DE_model.mutation_type],
                        'lowerF':[lowerF], 'upperF':[upperF],'tol':[tol],'F_delta':[F_delta], 'init':[str(init)], 
                        'refine_param':[str(DE_model.refine_param)], 'F_refine':[str(F_refine)], 'mutation_refine':[str(mutation_refine)], 
                        'lowerCR':[lowerCR], 'upperCR':[upperCR], 'CR_refine':[str(CR_refine)], 'CR_delta':[CR_delta],
                        'residual':[train_residual], 'run_avg_residual':[train_run_avg_residual_rmse], 'track_len':[track_len],
                        'return_method':[return_method], 'mutation_type_':[mutation_W0], 
                        'error_metric':[error_metric], 'reg_flag':[reg_flag], 
                        'run_enh':[str(run_enh)], 'bootstrapping':[str(DE_model.bootstrapping)],
                        'train_size':[str(train_size_)], 'current':[current], 'i_accept':[i_accept], 
                        'TrainRMSE':[min_value],'exh':[str(DE_model.exhaustive)], 'val_sample':[val_sample], 
                        # 'ValScore':[gen_val_score], 'ValResidual':[val_residual], 'ValRAResid':[val_run_avg_resid], 
                        'clustering_score':[c_min_value], 'local_score':[l_fit], 
                        'svd_value':[svd_fit], 's_scalar_value':[s_scalar_value], 's_exp_value':[s_exp_value], 
                        'Exit':[str(False)],
                    })        
        
        logging.info(f'run {run} gen {i} index {mindex} {error_metric} {min_value} train resid {train_residual} val resid {val_residual} current {current}')
        
        if i == G-1:
            logging.info(f'run {run} gen {i} maximum generation exit criteria')
            df['Exit'] = str(True)
        
        exit_criteria = val_run_avg_resid > 0 and i > val_gen_min and run_val and i_accept == 0
        
        if not exit_criteria:
            df_list.append(df)

        if exit_criteria:
            logging.info(f'run {run} gen {i} validation fitness exit criteria')
            df['Exit'] = str(True)
            df_list.append(df)
            break     

    if run_mcmc:
        mcmc_chain = W0, W1, W2, W3, b0, b1, b2, b3
        
    if not run_mcmc:    
        mcmc_chain = None

    #optimum_point = xgp[mindex]
    gen_points = xgp
    mcmc_chain = chain_vector

    optimum_point = split_candidate(xgp[mindex], m, n1, n2, n3)

    x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3  = {}, {}, {}, {}, {}, {}, {}, {}

    for j in NP_indices:
        candidates = split_candidate(ix_[j], m, n1, n2, n3)
        W0, W1, W2, W3, b0, b1, b2, b3 = candidates
        x_W0[j], x_W1[j], x_W2[j], x_W3[j], x_b0[j], x_b1[j], x_b2[j], x_b3[j] = W0, W1, W2, W3, b0, b1, b2, b3

    gen_points = x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3 
    val_points = gen_points
    dfs = pd.concat(df_list, sort = False)
    dfs['AcceptanceRate'] = np.sum(dfs.Acceptance)/G

    blah = pd.DataFrame(dfs, columns = ['current'])
    blah = blah[blah['current'] > 0].copy()
    dfs['StagnationPerc'] = len(blah)/G

    boo = pd.DataFrame(dfs, columns = ['current'])
    boo['test'] = boo['current'].shift(-1)
    boo.loc[(boo['current'] > 0) & (boo['test'] == 0), 'StagnationLen'] = True
    boo = boo[boo['StagnationLen'] == True].copy()
    stagnation_mean = np.mean(boo.current)
    dfs['MeanStagLen'] = stagnation_mean

    return optimum_point, gen_points, val_points, dfs

