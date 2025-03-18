import pandas as pd
import ray
from scipy.special import expit
import itertools
import os
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
from pandas_ods_reader import read_ods
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_log_error, r2_score, root_mean_squared_error
from sklearn.metrics import median_absolute_error, mean_pinball_loss
from scipy.special import expit
from scipy.linalg import svd
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import random_projection
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import iqr
import torch
import math
from scipy.spatial import distance_matrix
from itertools import permutations
from scipy.signal import find_peaks

np.random.seed(42)

class DEModelClass():
    
    def __init__(self, NP, g, F, CR, mutation_type, tol, NPI, init, track_length,
                    F_refine, F_delta, lowerF, upperF,
                    mutation_refine, refine_param, 
                    CR_refine, CR_delta, lowerCR, upperCR,
                    return_method, error_metric, run_enh, 
                    run, d, test_function):
        
        self.NP = NP
        self.g = g
        self.F = F
        self.CR = CR
        self.d = d
        self.dir_path = r'/home/wesley/repos/data'

        self.mutation_type = mutation_type
        self.F_delta = F_delta
        self.tol = tol
        self.NPI = NPI
        self.init = init
        self.lowerF = lowerF
        self.upperF = upperF
        self.track_length = track_length
        self.refine_param = refine_param
        self.F_refine = F_refine        
        self.mutation_refine = mutation_refine

        self.lowerCR = lowerCR
        self.upperCR = upperCR
        self.CR_refine = CR_refine
        self.CR_delta = CR_delta
       
        self.return_method = return_method        
        self.error_metric = error_metric
        self.run_enh = run_enh
        self.return_F_CR = return_F_CR
        self.return_mutation_list = return_mutation_list
        self.return_mutation_type = return_mutation_type
        self.return_running_avg_residual = return_running_avg_residual
        self.perform_svd_filter = perform_svd_filter
        self.perform_svd_scalar = perform_svd_scalar
        self.perform_clustering = perform_clustering
        self.perform_search = perform_search
        self.run = run
        self.test_function = test_function

        if self.test_function in ['rosenbrock']:
            self.analytical = rosenbrock_eval


def rosenbrock_eval(p, d):
    # candidate vector wise    
    x_i = p[0:d-1,:] # range is 1 to d-1
    x_pi = p[1:d,:] # range is 2 to d

    a = (x_i-1)**2
    b = 100*(x_pi - x_i**2)**2
    c = a + b
    f = np.sum(c, axis=0)
    return f

def return_error_metric(y_, yp, error_metric, weights):

    y_ = np.array(y_, dtype=np.float64)
    yp = np.array(yp, dtype=np.float64)

    size = len(y_)
    
    if error_metric == 'rmse':
        score = root_mean_squared_error(y_, yp, sample_weight=weights)

    if error_metric == 'mse':
        score = root_mean_squared_error(y_, yp, squared=True, sample_weight=weights)

    if error_metric == 'mae':
        score = mean_absolute_error(y_, yp, sample_weight=weights)

    if error_metric == 'mape':
        score = mean_absolute_percentage_error(y_, yp, sample_weight=weights)

    if error_metric == 'rmsle':
        try:
            score = root_mean_squared_log_error(y_, yp, sample_weight=weights)
        except:
            score = root_mean_squared_log_error(np.abs(y_), yp, squared=False, sample_weight=weights)
    
    if error_metric == 'msle':
        score = mean_squared_log_error(y_, yp, squared=True, sample_weight=weights)


    if error_metric == 'med_abs':
        score = median_absolute_error(y_, yp)

    if error_metric == 'pinball':
        score = mean_pinball_loss(y_, yp, alpha=0.1)

    if error_metric == 'r2':
        score = 1-r2_score(y_, yp)

    return score


def elu(w, alpha):
  """Exponential Linear Unit (ELU) activation function."""
  return np.where(w > 0, w, alpha * (expit(w) - 1))

def lrelu(w, alpha):
    w_ = np.maximum(alpha*w, w)
    return w_

def relu(w,dummy):
    # np.maximum is much faster than masking
    w_ = np.maximum(0,w)
    return w_

def vector_to_matrix(mu_vector, sigma_vector, a, b, key):
    # generate random matrices following normal distribution
    # specify mean of each component
    # vector input to matrix generation

    #mu = np.array([1,5,25,50])
    #sigma = np.array([1,5,25,50])
    #sigma_vector = np.abs(sigma_vector)

    matrix = np.array(
    [np.random.normal(m, s, 1) for m, s in zip(mu_vector, sigma_vector)]
    )
    matrix = matrix.reshape(a,b)
    return matrix


def perturbation_param(i, current):

    samples = 500
    low = -0.1
    high = 0.1

    return samples, low, high

def random_uniform(xgp, samples, NP_indices, DE_model):
    d = DE_model.d
    NP = DE_model.NP

    NP = len(NP_indices)
    total = samples*NP
    low_ = -1e-1
    high_ = 1e-1

    search_array = np.repeat(xgp, samples, axis=1)
    perturbation = np.random.uniform(low=low_, high=high_, size=(d, total))
    local_array = search_array + perturbation

    return local_array


def svd_space(M, alpha):

    # alpha = how many singular values to exclude

    U, S, V_T = svd(M)
    w0 = len(S)    
    Sigma = np.zeros((M.shape[0], M.shape[1]))

    S_ = np.diag(S)        
        
    #k = int(w0*alpha)
    S_[w0-alpha:] = 0
    Sigma[:w0, :w0] = S_
    M_ = U @ Sigma @ V_T
    return M_

def reconstruct_SVD(U, S, V_T):

    Sigma = np.zeros((U.shape[0], V_T.shape[1]))

    # populate Sigma with n x n diagonal matrix

    S_ = np.diag(S)
    w = len(S)
    Sigma[:w, :w] = S_
    M_ = U @ Sigma @ V_T
    return M_

def cluster_array(xgp, clustering_type, num_of_clusters, d):

    # dimensions

    n_neighbors_ = int(np.sqrt(d))
    X = xgp.T

    # predetermined number of clusters for kmeans, spectral

    if clustering_type == 'kmeans':

        kmeans = KMeans(n_clusters=num_of_clusters, n_init=10, random_state=42)
        kmeans.fit(X)        
        c_kmeans = kmeans.predict(X)
        centers = kmeans.cluster_centers_
        clabels = c_kmeans

    # spectral clustering n_neighbors = num_of_clusters,

    if clustering_type == 'spectral':
        sc = SpectralClustering(n_clusters=num_of_clusters, n_init=10, affinity='nearest_neighbors', 
                                n_neighbors = n_neighbors_, random_state=42).fit(X)

        # determine centers from clustering

        df = pd.DataFrame.from_dict({
                'id': list(sc.labels_) ,
                'data': list(X)
            })
        #centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).median().agg(np.array,1) original
        centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).mean().agg(np.array,1)
        centers = centers.reset_index(drop = True)
        centers = np.array([np.broadcast_to(row, shape=(d)) for row in centers])
        clabels = sc.labels_

    # affinity

    if clustering_type == 'affinity':
        ap = AffinityPropagation(random_state=42)
        c_means = ap.fit(X)
        centers = ap.cluster_centers_
        clabels = c_means    
    
    # agglomerative

    if clustering_type == 'agg':
        agg = AgglomerativeClustering(n_clusters=num_of_clusters)
        c_means = agg.fit_predict(X)
        clf = NearestCentroid()
        clf.fit(X, c_means)
        centers = clf.centroids_
        clabels = c_means

    # no predetermined number of clusters for mean shift, dbscan
    
    if clustering_type == 'mean_shift':
        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
        mean_shift = MeanShift(bandwidth=bandwidth, cluster_all = False)
        mean_shift.fit(X)
        c_means = mean_shift.predict(X)
        centers = mean_shift.cluster_centers_
        num_of_clusters = len(centers)
        clabels = c_means

    # dimensionality reduction, then use kmeans
    
    if clustering_type == 'rand_proj':
        transformer = random_projection.SparseRandomProjection(random_state=42, eps=0.75)
        X_new = transformer.fit_transform(xgp)
        print(X_new.shape)

    if clustering_type == 'nmf':
        nmf = NMF(n_components=2, init='random', random_state=42)
        X_new = nmf.fit_transform(xgp)
        print(X_new.shape)

    # convert center points array into dict
        
    centers = centers.T
    return centers

def return_F_CR(flag, lowerF, upperF, F_delta, F_, d, NP):

    if flag == 'default':
        F = np.full((d, NP), F_)

    if flag == 'variable':
        movie_list = np.arange(lowerF,upperF,F_delta)
        movie_list = np.round(movie_list,2)
        movie_list = list(movie_list)
        Fv = random.choice(movie_list)
        F = np.full((d, NP), Fv)
    
    if flag == 'dimension_variable':
        movie_list = np.arange(lowerF,upperF,F_delta)
        movie_list = np.round(movie_list,2)
        movie_list = list(movie_list)
        Fiv = random.choices(movie_list, k=d)
        Fiv = np.array(Fiv)
        Fiv = Fiv.reshape(len(Fiv),1)
        F = np.full((d, NP), Fiv)

    if flag == 'candidate_variable':
        movie_list = np.arange(lowerF,upperF,F_delta)
        movie_list = np.round(movie_list,2)
        movie_list = list(movie_list)
        Fiv = random.choices(movie_list, k=NP)
        Fiv = np.array(Fiv)
        F = np.full((d, NP), Fiv)
        boo=False

    if flag == 'full_variable':
        movie_list = np.arange(lowerF,upperF,F_delta)
        movie_list = np.round(movie_list,2)
        movie_list = list(movie_list)
        F = np.random.choice(movie_list, size=(d, NP))

    return F

def return_mutation_type(flag, mutation_list, mutation_default):

    if flag == 'default':
        mutation_ = mutation_default

    if flag == 'variable':
        mutation_ = random.choice(mutation_list)
    
    return mutation_


def return_mutation_list(NP):

    if NP >= 4:
        mutation_list = ['best', 'random']

    if NP >= 6:
        mutation_list = ['best', 'best2', 'random', 'random2',]

    if NP >= 8:
        mutation_list = ['best', 'best2', 'best3', 'random', 'random2', 'random3']

    return mutation_list


def return_running_avg_residual(i, value, gen_fitness_list, resid_tracking_list, track_len):
    gen_fitness_list.append(value)
    gen_train_residual = gen_fitness_list[i]-gen_fitness_list[i-1]
    resid_tracking_list.append(gen_train_residual)
    running_avg_residual = sum(resid_tracking_list[-track_len:])/len(resid_tracking_list[-track_len:])
    return gen_fitness_list, gen_train_residual, resid_tracking_list, running_avg_residual


def return_F(key, F_one, F_two, F_three):

    F_W0, F_W1, F_W2, F_W3, F_b0, F_b1, F_b2, F_b3 = F_one
    F2_W0, F2_W1, F2_W2, F2_W3, F2_b0, F2_b1, F2_b2, F2_b3 = F_two
    F3_W0, F3_W1, F3_W2, F3_W3, F3_b0, F3_b1, F3_b2, F3_b3 = F_three

    if key == 'W0':
        F_1 = F_W0
        F_2 = F2_W0
        F_3 = F3_W0

    if key == 'W1':
        F_1 = F_W1
        F_2 = F2_W1
        F_3 = F3_W1

    if key == 'W2':
        F_1 = F_W2
        F_2 = F2_W2
        F_3 = F3_W2

    if key == 'W3':
        F_1 = F_W3
        F_2 = F2_W3
        F_3 = F3_W3

    if key == 'b0':
        F_1 = F_b0
        F_2 = F2_b0
        F_3 = F3_b0

    if key == 'b1':
        F_1 = F_b1
        F_2 = F2_b1
        F_3 = F3_b1

    if key == 'b2':
        F_1 = F_b2
        F_2 = F2_b2
        F_3 = F3_b2

    if key == 'b3':
        F_1 = F_b3
        F_2 = F2_b3
        F_3 = F3_b3
    
    return F_1, F_2, F_3


def mutation_vector(NP, NP_indices, F_1, F_2, F_3, x_weight, MCMC, gen_best_x_weight, mutation_type):

    if mutation_type == 'random':

        y = mutate(NP, NP_indices, F_1, x_weight, MCMC)

    # DE/rand/2 needs minimum NP = 6

    if mutation_type == 'random2':

        y = mutate_two(NP, NP_indices, F_1, F_2, x_weight, MCMC)

    # DE/rand/3 needs minimum NP = 8

    if mutation_type == 'random3':

        y = mutate_three(NP, NP_indices, F_1, F_2, F_3, x_weight, MCMC)
    
    # DE/best/123

    if mutation_type in ['best']:

        y = mutate_best(NP, NP_indices, F_1, gen_best_x_weight, x_weight, MCMC)

    if mutation_type in ['best2']:

        y = mutate_best_two(NP, NP_indices, F_1, F_2, gen_best_x_weight, x_weight, MCMC)

    if mutation_type in ['best3']:

        y = mutate_best_three(NP, NP_indices, F_1, F_2, F_3, gen_best_x_weight, x_weight, MCMC)

    return y

def mutate(NP, NP_indices, F, x, MCMC):

    # mcmc noise addition

    b,c = x[0].shape

    # random mutation with distinct indices
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 3, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    y = x.copy()

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        base = x[i].copy() 
        v1 = x[j].copy() 
        v2 = x[k].copy() 
        p = base + F*(v2-v1) + generate_noise(b,c,MCMC)
        #p = base + F.dot(v2-v1) + generate_noise(b,c,MCMC)
        y[e] = p

    return y

def mutate_best(NP, NP_indices, F, gen_best, x, MCMC):
    
    # mcmc noise addition

    b,c = x[0].shape

    # best mutation with distinct indices for each index
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 2, replace=False)
        a = list(a)
        a.insert(0, j )
        test.append(a)
        indices = list(np.arange(0,NP))

    y = x.copy()

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        base = gen_best.copy()
        v1 = x[j].copy()
        v2 = x[k].copy()
        p = base + F*(v2-v1) + generate_noise(b,c,MCMC)
        #p = base + F.dot(v2-v1) + generate_noise(b,c,MCMC)
        y[e] = p

    return y

def mutate_best_three(NP, NP_indices, F, F2, F3, gen_best, x, MCMC):
    
    # mcmc noise addition

    b,c = x[0].shape

    # random mutation with distinct indices
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 6, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    y = x.copy()

    for e in NP_indices:
        j = test[e][0]
        k = test[e][1]
        l = test[e][2]
        m = test[e][3]
        n = test[e][4]
        o = test[e][5]
        base = gen_best
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]
        v5 = x[n]
        v6 = x[o]
        p = base + F*(v2-v1) + F2*(v4-v3) + F3*(v6-v5) + generate_noise(b,c,MCMC)
        #p = base + F.dot(v2-v1) + F2.dot(v4-v3) + F3.dot(v6-v5) + generate_noise(b,c,MCMC)
        y[e] = p

    return y

def mutate_two(NP, NP_indices, F, F2, x, MCMC):
    
    # mcmc noise addition

    b,c = x[0].shape

    # random mutation with distinct indices
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 5, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    y = x.copy()

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        l = test[e][3]
        m = test[e][4]
        base = x[i]
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]
        p = base + F*(v2-v1) + F2*(v4-v3) + generate_noise(b,c,MCMC)
        #p = base + F.dot(v2-v1) + F2.dot(v4-v3) + generate_noise(b,c,MCMC)
        y[e] = p

    return y

def mutate_best_two(NP, NP_indices, F, F2, gen_best, x, MCMC):
    
    # mcmc noise addition

    b,c = x[0].shape

    # random mutation with distinct indices
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 4, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    y = x.copy()

    for e in NP_indices:
        j = test[e][0]
        k = test[e][1]
        l = test[e][2]
        m = test[e][3]
        base = gen_best
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]
        p = base + F*(v2-v1) + F2*(v4-v3) + generate_noise(b,c,MCMC)
        #p = base + F.dot(v2-v1) + F2.dot(v4-v3) + generate_noise(b,c,MCMC)
        y[e] = p

    return y


def mutate_three(NP, NP_indices, F, F2, F3, x, MCMC):
    
    # mcmc noise addition

    b,c = x[0].shape

    # random mutation with distinct indices
    
    indices = list(np.arange(0,NP))
    test=[]

    for j in NP_indices:
        indices.remove(j)
        a = np.random.choice(indices, 7, replace=False)
        a = list(a)
        test.append(a)
        indices = list(np.arange(0,NP))

    y = x.copy()

    for e in NP_indices:
        i = test[e][0]
        j = test[e][1]
        k = test[e][2]
        l = test[e][3]
        m = test[e][4]
        n = test[e][5]
        o = test[e][6]
        base = x[i]
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]
        v5 = x[n]
        v6 = x[o]
        p = base + F*(v2-v1) + F2*(v4-v3) + F3*(v5-v6) + generate_noise(b,c,MCMC)
        #p = base + F.dot(v2-v1) + F2.dot(v4-v3) + F3.dot(v5-v6) + generate_noise(b,c,MCMC)
        y[e] = p

    return y


# def crossover_component(NP_indices, y, x, CR, key):

#     # crossover here is component-wise
#     # at least one component per row swapped based on random int k

#     z = x.copy()

#     if key in ['W0','W1','W2',]:
#         for e in NP_indices:
#             x_ = x[e].copy()
#             y_ = y[e].copy()
#             z_ = z[e].copy()

#             m,n = x_.shape

#             for i in np.arange(0,m):
#                 k = np.random.choice(np.arange(0,n),) # think this should be n
#                 for j in np.arange(0,n):
#                     if (random.uniform(0, 1) <= CR or j == k): # think this should be j
#                         z_[i,j] = y_[i,j].copy()
#                     else:
#                         z_[i,j] = x_[i,j].copy()
#             z[e] = z_.copy()

#     if key in ['W3', 'b0','b1','b2', 'b3']:
#         for e in NP_indices:
#             x_ = x[e].copy()
#             y_ = y[e].copy()
#             z_ = z[e].copy()

#             m,n = x_.shape

#             for i in np.arange(0,m):
#                 k = np.random.choice(np.arange(0,n),) # think this should be n
#                 for j in np.arange(0,n):
#                     if (random.uniform(0, 1) <= CR or j == k): # think this should be j
#                         z_[i,j] = y_[i,j].copy()
#                     else:
#                         z_[i,j] = x_[i,j].copy()
#             z[e] = z_.copy()
#     return z


def return_mutation_current(NP, mutation_type):

    if mutation_type in ['random', 'best']:
        k = 3
    if mutation_type in ['random2', 'best2']:
        k = 5
    if mutation_type in ['random3', 'best3']:
        k = 7
        
    permutations = math.perm(NP-1, k)

    return permutations

def return_combo_list(functions, r):    
    master = []        
    l = list(itertools.product(functions, repeat=r))    
    return l

def return_refine_count(df):
    cols = ['ClusterCount', 'LocalCount', 'n_SVD_Count', 'scalar_SVD_Count', 'exp_SVD_Count']
    df.loc[(df['clustering_score'] > 0) & (df['clustering_score'] <= df['TrainMin']), 'ClusterCount'] = 1
    df.loc[(df['local_score'] > 0) & (df['local_score'] <= df['TrainMin']), 'LocalCount'] = 1

    df.loc[(df['svd_value'] > 0) & (df['svd_value'] <= df['TrainMin']), 'n_SVD_Count'] = 1
    df.loc[(df['s_scalar_value'] > 0) & (df['s_scalar_value'] <= df['TrainMin']), 'scalar_SVD_Count'] = 1
    df.loc[(df['s_exp_value'] > 0) & (df['s_exp_value'] <= df['TrainMin']), 'exp_SVD_Count'] = 1

    final = df.groupby(['Run', 'NP'])[cols].sum()
    return final

def return_standard(return_method, dfs, optimum_point, error_metric,models, print_master, DE_model):
    logging.info(f'starting {return_method}')
    top = [1]
    data = dfs[dfs['Exit'] == 'True'].copy()
    optimum_point = optimum_point.reshape(len(optimum_point),1)
    xgen_fitness = DE_model.analytical(optimum_point,DE_model.d)
    data['c'] = 1
    data['Minimum'] = xgen_fitness
    data['Point'] = [optimum_point.T]
    models.append(data)

    return models, data


def perform_clustering(NP, test_value, maindex, DE_model, gen_points, i):
    
    clustering_list = ['kmeans', 'spectral', 'agg']
    clustering_type = random.choice(clustering_list)

    num_of_clusters_list = list(np.arange(2,NP-2))
    if NP == 4:
        num_of_clusters_list = [3]
    num_of_clusters = random.choice(num_of_clusters_list)
    
    cgp_W0 = cluster_array(gen_points, clustering_type, num_of_clusters, DE_model.d)

    # find best cluster fitness

    cluster_fitness = DE_model.analytical(cgp_W0, DE_model.d)

    c_min_value = np.amin(cluster_fitness)
    c_index = np.where(cluster_fitness == c_min_value)
    c_index = c_index[0][0]

    if c_min_value < test_value:
        logging.info(f'gen {i} {clustering_type} {num_of_clusters} clustering min {c_min_value} min {test_value}')
        gen_points[:,maindex] = cgp_W0[:,c_index].copy()
        test_value = c_min_value
    
    return gen_points, test_value, c_min_value

def perform_search(NP, test_value, maindex, DE_model, gen_points,i, NP_indices, current):
    local_ = 20
    samples = local_ * (int(current/1000) + 1)
    
    local = random_uniform(gen_points, samples, NP_indices, DE_model)    
    search_fitness = DE_model.analytical(local, DE_model.d)
    
    # determine best search point

    l_fit = np.amin(search_fitness)
    mindex = np.where(search_fitness == l_fit)
    mindex = mindex[0] # index integer

    if l_fit < test_value:
        logging.info(f'gen {i} local search min {l_fit} min {test_value}')
        gen_points[:,maindex] = local[:,mindex].reshape(DE_model.d,)
        test_value = l_fit

    return gen_points, test_value, l_fit


def perform_svd_filter(NP, test_value, maindex, DE_model, gen_points,i, NP_indices, current):
    j = 2
    svd_points= svd_space(gen_points, j)
    svd_fitness = DE_model.analytical(svd_points,DE_model.d)

    # determine best svd point

    svd_fit = np.amin(svd_fitness)
    mindex = np.where(svd_fitness == svd_fit)
    mindex = mindex[0] # index integer

    if len(mindex) > 1:
        mindex = mindex[0]

    if svd_fit < test_value:
        logging.info(f'gen {i} svd filter min {svd_fit} min {test_value}')
        gen_points[:,maindex] = svd_points[:,mindex].reshape(DE_model.d,)
        test_value = svd_fit

    return gen_points, test_value, svd_fit


def perform_svd_scalar(NP, test_value, maindex, DE_model, gen_points,i, NP_indices, current):

    points = gen_points 

    U_W0, S_W0, V_T_W0 = svd(gen_points)

    scalars = np.arange(0.1,2,0.1)
    scalars = np.delete(scalars, [7,8,9,10,11])

    for j in scalars:
        test = reconstruct_SVD(U_W0, S_W0*j, V_T_W0)
        points = np.hstack((points,test))        

    svd_scalar_fitness = DE_model.analytical(points,DE_model.d)
    s_scalar_value = np.amin(svd_scalar_fitness)
    mindex = np.where(svd_scalar_fitness == s_scalar_value)
    mindex = mindex[0] # index integer

    if len(mindex) > 1:
        mindex = mindex[0]

    if s_scalar_value < test_value:
        logging.info(f'gen {i} svd scalar {s_scalar_value} min {test_value}')
        gen_points[:,maindex] = points[:,mindex].reshape(DE_model.d,)
        test_value = s_scalar_value

    return gen_points, test_value, s_scalar_value


def post_DE(post_de_args, de_output):
    optimum_point, gen_points, dfs = de_output
    error_metric, models,DE_model, NP_indices, return_method, print_master, NP, DE_model = post_de_args

    # it looks like standard with bootstrapping=True is bumping

    if DE_model.return_method in ['standard', 'standard_val']:
        models, data = return_standard(return_method, dfs, optimum_point, error_metric,models, print_master, DE_model)
    
    return models, data
