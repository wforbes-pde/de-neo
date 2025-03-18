import numpy as np
import pandas as pd
import random
import logging
from numpy import array
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import qmc
from pyDOE import lhs
from scipy.linalg import svd

np.random.seed(42)
print_master = True

# https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube
def lhs_init(samples, dim, bounds):

    # appears to be a samples of R^b
    # is this okay? 

    sample = lhs(n=dim, samples=samples)
    pop = np.zeros((samples, dim))
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
        a = np.random.choice(indices, 7, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        l = test[e][3]
        m = test[e][4]
        n = test[e][5]
        o = test[e][6]
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
        a = np.random.choice(indices, 2, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    for e in NP_indices:
        j = test[e][0]
        k = test[e][1]
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
        a = np.random.choice(indices, 4, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    for e in NP_indices:
        j = test[e][0]
        k = test[e][1]
        l = test[e][2]
        m = test[e][3]
        base_array[:,e] = gen_best
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
        a = np.random.choice(indices, 6, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    for e in NP_indices:
        j = test[e][0]
        k = test[e][1]
        l = test[e][2]
        m = test[e][3]
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


def selection_vector(d, NP_indices, DE_model, i, mindex, error_metric, x_points, z_points):
    
    # determine survival of target or trial vector
    # into the next generation

    x_points_orig = x_points.copy()

    x_gen_fitness = DE_model.analytical(x_points,d)
    z_gen_fitness = DE_model.analytical(z_points,d)

    index_lower = np.where(z_gen_fitness < x_gen_fitness)    
    x_points[:,index_lower] = z_points[:,index_lower]

    i_accept = len(index_lower)

    return x_points, i_accept

def crossover_vector(y, x, CR):

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


def generate_initial_population(d, NP, NP_indices, init):

    # NP number of candidates
    
    # init args
    
    itype,l,h = init 

    if itype == 'uniform':
        x =np.random.uniform(low=l, high=h, size=(d, NP))

    # loc = mean, scale = standard deviation   

    if itype == 'normal':
        mean_ = l
        sigma = h
        x = np.random.normal(loc=mean_, scale=sigma, size=(d,NP))

    if itype == 'latin':
        bounds=h
        x = lhs_init(samples=NP, dim=d, bounds=bounds)
        x = x.T

    if itype == 'halton':
        sampler = qmc.Halton(d=d, scramble=True)
        x = sampler.random(n=NP)
        x = x.T
    
    return x

def differential_evolution_vector(DE_model):
    
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
            
            if init[0] in ['normal', 'uniform', 'halton', 'latin']:

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
                
                variation_list = ['default', 'variable', 'dimension_variable', 'candidate_variable', 'full_variable',]
                F_refine = random.choice(variation_list)
                CR_refine = random.choice(variation_list)
                mutation_refine = random.choice(variation_list)

            if F_refine == 'variable' and current > refine_current_start and i > refine_gen_start:
                F_array = DE_model.return_F_CR('variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)
                F2_array = DE_model.return_F_CR('variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)
                F3_array = DE_model.return_F_CR('variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)

            if F_refine == 'dimension_variable' and current > refine_current_start and i > refine_gen_start:
                F_array = DE_model.return_F_CR('dimension_variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)
                F2_array = DE_model.return_F_CR('dimension_variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)
                F3_array = DE_model.return_F_CR('dimension_variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)

            if F_refine == 'candidate_variable' and current > refine_current_start and i > refine_gen_start:
                F_array = DE_model.return_F_CR('candidate_variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)
                F2_array = DE_model.return_F_CR('candidate_variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)
                F3_array = DE_model.return_F_CR('candidate_variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)

            if F_refine == 'full_variable' and current > refine_current_start and i > refine_gen_start:
                F_array = DE_model.return_F_CR('full_variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)
                F2_array = DE_model.return_F_CR('full_variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)
                F3_array = DE_model.return_F_CR('full_variable', lowerF, upperF, CR_delta, DE_model.F, d, NP)

            if CR_refine == 'variable' and current > refine_current_start and i > refine_gen_start:
                CR_array = DE_model.return_F_CR('variable', lowerF, upperF, CR_delta, DE_model.CR, d, NP)

            if mutation_refine == 'variable' and current > refine_current_start and i > refine_gen_start:                
                mutation_op = DE_model.return_mutation_type('variable', mutation_list, DE_model.mutation_type)           

        
        # mutation

        y = mutation_vector(d, NP, NP_indices, F_array, F2_array, F3_array, x, gen_best_x, mutation_op)
                
        # crossover

        z = crossover_vector(y, x, CR_array)
        
        # selection

        xgp, i_accept = selection_vector(d, NP_indices, DE_model, i, mindex, DE_model.error_metric, x, z)
        
        accept_list.append(i_accept)

        # fitness evaluation

        gen_fitness_values = DE_model.analytical(x,d)

        # determine best generation point

        min_value = np.amin(gen_fitness_values)
        mindex = np.where(gen_fitness_values == min_value)
        mindex = mindex[0][0] # index integer

        # determine worst generation point

        max_value = np.amax(gen_fitness_values)
        maindex = np.where(gen_fitness_values == max_value )
        maindex = maindex[0][0]

        # define generation best

        gb_x = xgp[:,mindex]

        # training residual tracking

        gen_train_fitness_list, train_residual, gen_train_resid_list, train_run_avg_residual_rmse = \
            DE_model.return_running_avg_residual(i, min_value, gen_train_fitness_list, gen_train_resid_list, track_len)
        
        if train_residual > 0:
            logging.info(f'run {run} gen {i} index {mindex} minimum {min_value} train resid {train_residual} val resid {val_residual} current {current}')
            #breakpoint()

        # refinement

        comparison_value = min_value
        c_min_value = 1e6
        l_fit = 1e6
        svd_fit = 1e6
        s_scalar_value = 1e6
        s_exp_value = 1e6

        # clustering

        if current > refine_current_start and i_accept > 0 and run_cluster and current % refine_mod_start in cluster_r:
            cluster_points, comparison_value, c_min_value = DE_model.perform_clustering(NP, comparison_value, maindex, DE_model, xgp,i)
            xgp = cluster_points

        # local search 

        if current > refine_current_start and i_accept > 0 and run_local and current % refine_mod_start in local_r:
            search_points, comparison_value, l_fit = DE_model.perform_search(NP,comparison_value, maindex, DE_model, xgp,i, NP_indices, current)
            xgp = search_points

        # SVD filter

        if current > refine_current_start and i_accept > 0 and run_svd and current % refine_mod_start in svd_filter_r:
            svd_points, comparison_value, svd_fit = DE_model.perform_svd_filter(NP,comparison_value, maindex, DE_model, xgp,i, NP_indices, current)
            xgp = svd_points
        
        # scalar SVD

        if current > refine_current_start and i_accept > 0 and run_svd and current % refine_mod_start in svd_scalar_r:
            svd_scalar_points, comparison_value, s_scalar_value = DE_model.perform_svd_scalar(NP, comparison_value, maindex, DE_model, xgp,i, NP_indices, current)
            xgp = svd_scalar_points
        
        # # exp scalar

        # if current > refine_current_start and i_accept > 0 and run_svd and current % refine_mod_start in svd_exp_r:
        #     svd_exp_points, s_exp_value  = DE_model.perform_svd_exp(NP, comparison_value, maindex, DE_model, gen_points,i, NP_indices, current)
        #     xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = svd_exp_points
        
        # collect parameters and data
        
        if refine_random:
            F_refine = 'random'
            CR_refine = 'random'
            mutation_refine = 'random'
        
        df = pd.DataFrame({'Run':[run], 'Generation':[i], 'F':[F], 'CR':[CR_array[0,0]], 'G':[G], 'NP':[NP], 'NPI':[NPI],'mutation_type':[DE_model.mutation_type],
                        'lowerF':[lowerF], 'upperF':[upperF],'tol':[tol],'F_delta':[F_delta], 'init':[str(init)], 
                        'refine_param':[str(DE_model.refine_param)], 'F_refine':[str(F_refine)], 'mutation_refine':[str(mutation_refine)], 
                        'lowerCR':[lowerCR], 'upperCR':[upperCR], 'CR_refine':[str(CR_refine)], 'CR_delta':[CR_delta],
                        'residual':[train_residual], 'run_avg_residual':[train_run_avg_residual_rmse], 'track_len':[track_len],
                        'return_method':[return_method], 'mutation_type_':[mutation_op], 
                        'error_metric':[error_metric], 
                        'run_enh':[str(run_enh)], 
                         'current':[current], 'i_accept':[i_accept], 
                        'TrainMin':[min_value],'exh':[str(DE_model.exhaustive)], 'val_sample':[val_sample], 
                        # 'ValScore':[gen_val_score], 'ValResidual':[val_residual], 'ValRAResid':[val_run_avg_resid], 
                        'clustering_score':[c_min_value], 'local_score':[l_fit], 
                        'svd_value':[svd_fit], 's_scalar_value':[s_scalar_value], 's_exp_value':[s_exp_value], 
                        'Exit':[str(False)],
                    })        
        
        logging.info(f'run {run} gen {i} index {mindex} {error_metric} minimum {min_value} train resid {train_residual} val resid {val_residual} current {current}')
        
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

    #optimum_point = xgp[mindex]
    gen_points = xgp
    optimum_point = xgp[:,mindex]
    val_points = gen_points
    dfs = pd.concat(df_list, sort = False)

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

