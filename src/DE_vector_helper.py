import pandas as pd
#from math import comb
import ray
from scipy.special import expit
import itertools
import os
import sys
from datetime import datetime
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
from pandas_ods_reader import read_ods
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_log_error, r2_score, root_mean_squared_error
from sklearn.metrics import median_absolute_error, mean_pinball_loss
from scipy.special import expit
from scipy.linalg import svd
from fitter import Fitter
from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPRegressor
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import random_projection
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
from sklearn.model_selection import KFold
#from scipy.fftpack import fft, ifft
from scipy.stats import iqr
import torch
#from torchmetrics.regression import LogCoshError
import math
from scipy.spatial import distance_matrix
from itertools import permutations, combinations, combinations_with_replacement
from scipy.signal import find_peaks
from sklearn.neural_network import MLPRegressor
#from sklearn.decomposition import PCA, DictionaryLearning, FactorAnalysis, FastICA
#from astropy.visualization import hist
from sklearn.neighbors import KernelDensity
import seaborn as sn

np.random.seed(42)

class DEModelClass():
    
    def __init__(self, NP, g, F, CR, mutation_type, tol, NPI, init, track_length,
                    F_refine, F_delta, lowerF, upperF,
                    mutation_refine, refine_param, 
                    CR_refine, CR_delta, lowerCR, upperCR,
                    return_method, error_metric, run_enh, exhaustive, val_sample,
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
        self.exhaustive = exhaustive
        self.fitness = fitness
        self.return_F_CR = return_F_CR
        self.return_mutation_list = return_mutation_list
        self.return_mutation_type = return_mutation_type
        self.mutation = mutation
        self.crossover_component = crossover_component
        self.return_running_avg_residual = return_running_avg_residual
        self.perform_svd_filter = perform_svd_filter
        self.perform_svd_exp = perform_svd_exp
        self.perform_svd_log = perform_svd_log
        self.perform_clustering = perform_clustering
        self.perform_search = perform_search
        self.plot = plot
        self.skunk_initial = skunk_initial
        self.exhaustive_mutation = exhaustive_mutation
        self.val_sample = val_sample
        self.run = run
        self.test_function = test_function

        if self.test_function in ['rosenbrock']:
            self.analytical = rosenbrock_eval


class NNClass():
    
    def __init__(self, num_layers, n1, n2, n3, m, activation_, regularization_):
        
        activation, alpha = activation_
        reg_type, reg_alpha = regularization_

        self.num_layers = num_layers
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.activation = activation_
        self.activation_function = activation
        self.alpha = alpha
        self.regularization_ = regularization_
        self.reg_type = reg_type
        self.reg_alpha = reg_alpha

        # feature dimension

        self.m = m

        # activation function

        if activation == 'lrelu':
            self.activation_function = lrelu

        if activation == 'relu':
            self.activation_function = relu

        if activation == 'elu':
            self.activation_function = elu

        if activation == 'tanh':
            self.activation_function = np.tanh

        if activation == 'logistic':
            self.activation_function = expit
    
    def set_ytrain_std(self, std_):
        self.ytrain_std = std_

    def set_phase(self, phase):
        self.phase = phase


def rosenbrock(p, d): # used in selection operator
    # population component wise
    p = p.reshape(len(p),1)
    x_i = p[0:d-1,:] # range is 1 to d-1
    x_pi = p[1:d,:] # range is 2 to d

    a = (x_i-1)**2
    b = 100*(x_pi - x_i**2)**2
    c = a + b
    f = np.sum(c, axis=0)
    return f

def rosenbrock_eval(p, d):
    # candidate vector wise    
    x_i = p[0:d-1,:] # range is 1 to d-1
    x_pi = p[1:d,:] # range is 2 to d

    a = (x_i-1)**2
    b = 100*(x_pi - x_i**2)**2
    c = a + b
    f = np.sum(c, axis=0)
    return f


def fitness(x, W0, W1, W2, W3, b0, b1, b2, b3, y_, n_, error_metric, reg_flag, NN_model):
    
    reg1, reg2, reg3 = reg_flag
    # feed forward

    yp = feed_forward(x, W0, W1, W2, W3, b0, b1, b2, b3, n_, NN_model)
    yp_std = np.std(yp,axis=0)[0]
    
    # error function

    weights = None
    base_score = return_error_metric(y_, yp, error_metric, weights)

    # DE candidate regularization

    reg1_ratio = 0
    reg2_ratio = 0
    reg3_ratio = 0

    if reg1:
        train_std_threshold = 0.5 # 0.5 ??? was 0.1
        if yp_std < NN_model.ytrain_std*train_std_threshold:
            reg1_ratio = 0.2

    if reg2:
        threshold, scalar = (0.5,2) # ??? 0.2,2
        d = np.diff(yp, axis=0)
        e = d[ np.abs(d) <= threshold].copy()
        counts = len(e)
        zero_counts = np.count_nonzero(yp==0)
        penalty = (counts + zero_counts) * scalar
        reg2_ratio = penalty/n_ 

    if reg3:    
        yp1 = np.array(yp).ravel()
        peaks, _ = find_peaks(yp1, height=np.max(y_)*2)
        peaks_count = len(peaks)
        if peaks_count > 0:
            reg3_ratio = 1
    
    score = base_score * (1 + reg1_ratio + reg2_ratio + reg3_ratio)

    # add weight regularization to prevent overfitting

    if NN_model.phase == 'training':
        if NN_model.reg_type == 'l1':
            ralpha = 1
            reg = (np.linalg.norm(W0,ralpha) + np.linalg.norm(W1,ralpha) + np.linalg.norm(W2,ralpha) + np.linalg.norm(W3,ralpha)) * NN_model.reg_alpha
        if NN_model.reg_type == 'l2':
            ralpha = 2
            reg = (np.linalg.norm(W0,ralpha) + np.linalg.norm(W1,ralpha) + np.linalg.norm(W2,ralpha) + np.linalg.norm(W3,ralpha)) * NN_model.reg_alpha
        if NN_model.reg_type is None:
            reg = 0
        score = score + reg

    if NN_model.phase == 'testing':
        score = base_score
        
    return score, yp

def feed_forward(x, W0, W1, W2, W3, b0, b1, b2, b3, n_, NN_model):

    if NN_model.num_layers == 3:

        # convert bias vectors into bias matrices

        b0 = np.repeat(b0, n_, axis=0)
        b1 = np.repeat(b1, n_, axis=0)
        b2 = np.repeat(b2, n_, axis=0)
        b3 = np.repeat(b3, n_, axis=0)
    
        s1 = x@W0 + b0
        z1 = NN_model.activation_function(s1, NN_model.alpha)

        s2 = z1@W1 + b1
        z2 = NN_model.activation_function(s2, NN_model.alpha)

        s3 = z2@W2 + b2
        z3 = NN_model.activation_function(s3, NN_model.alpha)

        yp = NN_model.activation_function(z3@W3+b3, NN_model.alpha)

    return yp

def vary_error_weights(x, W0, W1, W2, W3, b0, b1, b2, b3, y_, 
                 n_, 
                 error_metric, reg_flag, error_weight, i,
                 NN_model):
    
    # convert bias vectors into bias matrices

    #W3 = np.repeat(W3, n_, axis=0)

    b0 = np.repeat(b0, n_, axis=0)
    b1 = np.repeat(b1, n_, axis=0)
    b2 = np.repeat(b2, n_, axis=0)
    b3 = np.repeat(b3, n_, axis=0)

    # b0 = np.repeat(b0, n_, axis=)
    # b1 = np.repeat(b1, n_, axis=1)
    # b2 = np.repeat(b2, n_, axis=1)
    # b3 = np.repeat(b3, n_, axis=1)

    # b0 = b0.T
    # b1 = b1.T
    # b2 = b2.T
    # b3 = b3.T
    
    # default weights
    # change with other conditions
    
    weights = None
    
    if error_weight is None:
        score = return_error_metric(y_, yp, error_metric, weights)

    # testing

    if error_weight is not None:
        keys = list(error_weight.keys())
        if len(keys) == 2:
            m1 = keys[0]
            w1 = error_weight[m1]

            m2 = keys[1]
            w2 = error_weight[m2]

            score1 = return_error_metric(y_, yp, m1, weights)
            score2 = return_error_metric(y_, yp, m2, weights)

            score = score1*w1 + score2*w2

        if len(keys) == 3:

            m1 = keys[0]
            w1 = error_weight[m1]

            m2 = keys[1]
            w2 = error_weight[m2]

            m3 = keys[2]
            w3 = error_weight[m3]

            score1 = return_error_metric(y_, yp, m1, weights)
            score2 = return_error_metric(y_, yp, m2, weights)
            score3 = return_error_metric(y_, yp, m3, weights)

            score = score1*w1 + score2*w2 + score3*w3
            #logging.info('score3')

        if len(keys) == 4:

            m1 = keys[0]
            w1 = error_weight[m1]

            m2 = keys[1]
            w2 = error_weight[m2]

            m3 = keys[2]
            w3 = error_weight[m3]

            m4 = keys[3]
            w4 = error_weight[m4]

            score1 = return_error_metric(y_, yp, m1, weights)
            score2 = return_error_metric(y_, yp, m2, weights)
            score3 = return_error_metric(y_, yp, m3, weights)
            score4 = return_error_metric(y_, yp, m4, weights)

            score = score1*w1 + score2*w2 + score3*w3 + score4*w4
            #logging.info('score4')

    return score, yp

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

    if error_metric == 'rae':
        score = c_error_metric.rae_loss(y_, yp, size)

    if error_metric == 'rse':
        score = c_error_metric.rse_loss(y_, yp, size)

    if error_metric == 'log_cosh':
        y_ = y_.reshape(len(y_),1)
        yp = torch.tensor(yp)
        y_ = torch.tensor(y_)
        score = log_cosh_loss(yp, y_)

    if error_metric == 'med_abs':
        score = median_absolute_error(y_, yp)

    if error_metric == 'pinball':
        score = mean_pinball_loss(y_, yp, alpha=0.1)

    if error_metric == 'r2':
        score = 1-r2_score(y_, yp)

    return score

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    
    val = torch.mean(_log_cosh(y_pred - y_true))
    return val.item()

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)

def log_cosh(true, pred):
    logcosh = np.log(np.cosh(pred - true))
    logcosh_loss = np.sum(logcosh)
    return logcosh_loss[0]

def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

def normalized_root_mean_squared_error(true, pred, weights):
    rmse = mean_squared_error(true, pred, squared=False, sample_weight=weights)    
    #nrmse_loss = rmse/np.std(pred)
    #nrmse_loss = rmse/np.mean(pred)
    #nrmse_loss = rmse/np.max(pred)
    #nrmse_loss = rmse/(np.max(pred)-np.min(pred))
    nrmse_loss = rmse/ iqr(pred)
    return nrmse_loss

def relative_absolute_error(true, pred):
    true_mean = np.mean(true)
    squared_error_num = np.sum(np.abs(true - pred))
    squared_error_den = np.sum(np.abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss

def relative_squared_error(true, pred):
    true_mean = np.mean(true)
    squared_error_num = np.sum(np.square(true - pred))
    squared_error_den = np.sum(np.square(true - true_mean))
    rse_loss = squared_error_num / squared_error_den
    return rse_loss


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

def plot_array(key, H):
     
    dir_path = '/home/wesley/repos/array'
    kfc = (key, ' Weight Std', )
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    plt.imshow(H, interpolation='none')
    #plt.close()
    plt.clf()

def plot_autocorrelation(key, p, q, test, parameter, run):

    #plt.figure(figsize=(18,6))
    lags_ = len(test)/4
    plot_acf(test, lags=lags_, markersize=1)
    
    #plt.xlabel('Lag')
    #plt.ylabel('AutoCorrelation')
    #plt.legend(fontsize = 10, loc='upper left')

    dir_path = '/home/wesley/repos/auto'
    kfc = (key, ' Chain Autocorrelation', str(p), str(q), ' Run', str(run))
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300)
    #plt.show()
    #plt.close('all')
    plt.clf()

def plot_weight_hist2(key, p, q, test, parameter, run, fig, bins):

    # std = np.std(test)
    # xmax = parameter + std
    # xmin = parameter - std

    # Normal Reference Rules

    if True:

        # draw histograms with two different bin widths
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)
        for i, bins in enumerate(['scott', 'freedman']):
            hist(test, bins=bins, ax=ax[i], histtype='stepfilled',
                alpha=0.2, density=True)
            ax[i].set_xlabel('t')
            ax[i].set_ylabel('P(t)')
            ax[i].set_title(f'hist(t, bins="{bins}")',
                            fontdict=dict(family='monospace'))
            
        # save image to file

        dir_path = '/home/wesley/repos/hist'
        kfc = (key, ' Matrix Hist Astro Normal', str(p), str(q), ' Run', str(run), ' Bins', str(bins))
        output_name = ' '.join(kfc)
        output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
        logging.info(f'Saving to {output_loc}')
        plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
        #plt.show()
        #plt.close('all')
        plt.clf()

    # Bayesian Models    

    if True:
    
        # draw histograms with two different bin widths
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)
        for i, bins in enumerate(['knuth', 'blocks']):
            hist(test, bins=bins, ax=ax[i], histtype='stepfilled',
                    alpha=0.2, density=True)
            ax[i].set_xlabel('t')
            ax[i].set_ylabel('P(t)')
            ax[i].set_title(f'hist(t, bins="{bins}")',
                            fontdict=dict(family='monospace'))

        # save image to file

        dir_path = '/home/wesley/repos/hist'
        kfc = (key, ' Matrix Hist Astro Bayes', str(p), str(q), ' Run', str(run), ' Bins', str(bins))
        output_name = ' '.join(kfc)
        output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
        logging.info(f'Saving to {output_loc}')
        plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
        #plt.show()
        #plt.close('all')
        plt.clf()

def plot_weight_hist(key, p, q, test, parameter, run, fig, bins_):

    #plt.figure(figsize=(12,6))
    std = np.std(test)
    xmax = parameter + std
    xmin = parameter - std
    #plt.hist(test, bins='auto', range=[xmin,xmax])
    plt.hist(test, bins=bins_, histtype='stepfilled') # histtype='stepfilled',
    plt.axvline(parameter, color='k', linestyle='dashed', linewidth=1)
    
    plt.xlabel('Value')
    plt.ylabel('Count')
    #plt.legend(fontsize = 10, loc='upper left')

    dir_path = '/home/wesley/repos/hist'
    kfc = (key, ' Matrix Hist', str(p), str(q), ' Run', str(run), ' Bins', str(bins_))
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    #plt.show()
    #plt.close('all')
    plt.clf()

def plot_weight_trace_plot(key, p, q, test, run, chain_slice):

    x_ = np.arange(0,len(test))
    #plt.figure(figsize=(12,6))
    plt.plot(x_, test,  label='chain', linewidth=0.5)
    
    plt.xlabel('Index')
    plt.ylabel('Value')
    #plt.legend(fontsize = 10, loc='upper left')

    dir_path = '/home/wesley/repos/trace'
    kfc = (key, ' Trace Plot', str(p), str(q), 'ChainSlice', str(chain_slice), ' Run', str(run))
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    plt.clf()
    #plt.close('all')

def plot_CI(dataset, xcol, samples_pred, application, daytype, run, mm):
    fig, ax = plt.subplots(figsize=(18,6)) # 18,6

    ax.plot(dataset[xcol], dataset['pred_mean'], label='Mean', linewidth=1, color='black')
    ax.plot(dataset[xcol], dataset['actual'], label='Actual', linewidth=1, color='red')
    ax.set_aspect('auto') 

    lower = np.percentile(samples_pred, 5, axis=0)
    upper = np.percentile(samples_pred, 95, axis=0)
    lower=lower.reshape(len(lower),)
    upper=upper.reshape(len(upper),)
    plt.fill_between(dataset['datetime'], dataset['lower'], dataset['upper'], alpha=0.5, label='5% and 95% confidence interval', color='cornflowerblue')
    plt.legend()
    plt.margins(x=0)
    plt.tight_layout()
    plt.xlabel('datetime')
    plt.ylabel('kWh')

    dir_path = '/home/wesley/repos/'
    kfc = ('Credible Interval', mm, application, str(daytype), 'Run', str(run))
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300)
    plt.clf()


def perturbation_param(i, current):

    samples = 500
    low = -0.1
    high = 0.1

    return samples, low, high

def random_uniform(xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3, samples, NP_indices):

    rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3 = {}, {}, {}, {}, {}, {}, {}, {}
    pgp_W0, pgp_W1, pgp_W2, pgp_W3, pgp_b0, pgp_b1, pgp_b2, pgp_b3 = {}, {}, {}, {}, {}, {}, {}, {}

    NP = len(NP_indices)
    total = samples*NP
    low_ = -1e-1
    high_ = 1e-1

    for e in np.arange(0,total):
        h=e%NP
        m,n = xgp_W0[0].shape
        X_W0 = np.random.uniform(low=low_, high=high_, size=(m,n))
        pgp_W0[e] = X_W0
        rgp_W0[e] = xgp_W0[h].copy() + X_W0

        m,n = xgp_W1[0].shape
        X_W1 = np.random.uniform(low=low_, high=high_, size=(m,n))
        pgp_W1[e] = X_W1
        rgp_W1[e] = xgp_W1[h].copy() + X_W1

        m,n = xgp_W2[0].shape
        X_W2 = np.random.uniform(low=low_, high=high_, size=(m,n))
        pgp_W2[e] = X_W2
        rgp_W2[e] = xgp_W2[h].copy() + X_W2

        m,n = xgp_W3[0].shape
        X_W3 = np.random.uniform(low=low_, high=high_, size=(m,n))
        pgp_W3[e] = X_W3
        rgp_W3[e] = xgp_W3[h].copy() + X_W3

        m,n = xgp_b0[0].shape
        X_b0 = np.random.uniform(low=low_, high=high_, size=(m,n))
        pgp_b0[e] = X_b0
        rgp_b0[e] = xgp_b0[h].copy() + X_b0

        m,n = xgp_b1[0].shape
        X_b1 = np.random.uniform(low=low_, high=high_, size=(m,n))
        pgp_b1[e] = X_b1
        rgp_b1[e] = xgp_b1[h].copy() + X_b1

        m,n = xgp_b2[0].shape
        X_b2 = np.random.uniform(low=low_, high=high_, size=(m,n))
        pgp_b2[e] = X_b2
        rgp_b2[e] = xgp_b2[h].copy() + X_b2

        m,n = xgp_b3[0].shape
        X_b3 = np.random.uniform(low=low_, high=high_, size=(m,n))
        pgp_b3[e] = X_b3
        rgp_b3[e] = xgp_b3[h].copy() + X_b3

    local = rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3

    return local

def random_uniform_delta(rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3, samples, NP_indices,
                   delta_X, e):

    lgp_W0, lgp_W1, lgp_W2, lgp_W3, lgp_b0, lgp_b1, lgp_b2, lgp_b3 = {}, {}, {}, {}, {}, {}, {}, {}
    X_W0, X_W1, X_W2, X_W3, X_b0, X_b1, X_b2, X_b3 = delta_X

    NP = len(NP_indices)
    total = samples*NP
    low_ = -1e-1
    high_ = 1e-1

    for w in np.arange(0,total):
        #h=w%NP
        m,n = rgp_W0[0].shape
        X_W0 = np.random.uniform(low=low_, high=high_, size=(m,n))
        lgp_W0[w] = rgp_W0[e].copy() + X_W0

        m,n = rgp_W1[0].shape
        lgp_W1[w] = rgp_W1[e].copy() + X_W1

        m,n = rgp_W2[0].shape
        lgp_W2[w] = rgp_W2[e].copy() + X_W2

        m,n = rgp_W3[0].shape
        lgp_W3[w] = rgp_W3[e].copy() + X_W3

        m,n = rgp_b0[0].shape
        lgp_b0[w] = rgp_b0[e].copy() + X_b0

        m,n = rgp_b1[0].shape
        lgp_b1[w] = rgp_b1[e].copy() + X_b1

        m,n = rgp_b2[0].shape
        lgp_b2[w] = rgp_b2[e].copy() + X_b2

        m,n = rgp_b3[0].shape
        lgp_b3[w] = rgp_b3[e].copy() + X_b3

    local = lgp_W0, lgp_W1, lgp_W2, lgp_W3, lgp_b0, lgp_b1, lgp_b2, lgp_b3 

    return local

def random_uniform_new(xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3, samples, NP_indices,
                       x_dict, y_dict, DE_model, reg_flag, n_,
                       gen_fitness, skunk_start, skunk_mod, bootstrapping, NN_model):

    rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3 = {}, {}, {}, {}, {}, {}, {}, {}
    pgp_W0, pgp_W1, pgp_W2, pgp_W3, pgp_b0, pgp_b1, pgp_b2, pgp_b3 = {}, {}, {}, {}, {}, {}, {}, {}

    NP = len(NP_indices)
    total = samples*NP
    low_ = skunk_start
    high_ = skunk_mod

    locale = None
        
    if not bootstrapping:
        n_ = len(y_dict)
        X_train = x_dict
        y_train = y_dict
    
    # perturbation around candidates

    for e in np.arange(0,total):
        h=e%NP
        m,n = xgp_W0[0].shape
        X_W0 = np.random.uniform(low=low_, high=high_, size=(m,n))        
        rgp_W0 = xgp_W0[h].copy() + X_W0

        m,n = xgp_W1[0].shape
        X_W1 = np.random.uniform(low=low_, high=high_, size=(m,n))        
        rgp_W1 = xgp_W1[h].copy() + X_W1

        m,n = xgp_W2[0].shape
        X_W2 = np.random.uniform(low=low_, high=high_, size=(m,n))        
        rgp_W2 = xgp_W2[h].copy() + X_W2

        m,n = xgp_W3[0].shape
        X_W3 = np.random.uniform(low=low_, high=high_, size=(m,n))        
        rgp_W3 = xgp_W3[h].copy() + X_W3

        m,n = xgp_b0[0].shape
        X_b0 = np.random.uniform(low=low_, high=high_, size=(m,n))        
        rgp_b0 = xgp_b0[h].copy() + X_b0

        m,n = xgp_b1[0].shape
        X_b1 = np.random.uniform(low=low_, high=high_, size=(m,n))        
        rgp_b1 = xgp_b1[h].copy() + X_b1

        m,n = xgp_b2[0].shape
        X_b2 = np.random.uniform(low=low_, high=high_, size=(m,n))        
        rgp_b2 = xgp_b2[h].copy() + X_b2

        m,n = xgp_b3[0].shape
        X_b3 = np.random.uniform(low=low_, high=high_, size=(m,n))
        rgp_b3 = xgp_b3[h].copy() + X_b3
    
        l_errors = []
        
        if bootstrapping:
            X_train = x_dict[h]
            y_train = y_dict[h]
            n_ = len(y_train)

        l_fit, rv = fitness(X_train, rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3,
                                y_train, n_, DE_model.error_metric, reg_flag, NN_model)
        l_errors.append(l_fit)        
        
        # perturbation around perturbed candidates based on fitness
        
        delta_X = X_W0, X_W1, X_W2, X_W3, X_b0, X_b1, X_b2, X_b3
        
        if l_fit < gen_fitness[h]:
            logging.info(f'l_fit {l_fit} gen fit {gen_fitness[h]} index {h}')
            gen_fitness[h] = l_fit
            xgp_W0[h], xgp_W1[h], xgp_W2[h], xgp_W3[h], xgp_b0[h], xgp_b1[h], xgp_b2[h], xgp_b3[h] = rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3
        
    pert = xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3,
    
    return pert


def random_normal(xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3, samples, NP_indices):

    rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3 = {}, {}, {}, {}, {}, {}, {}, {}

    NP = len(NP_indices)
    total = samples*NP

    for e in np.arange(0,total):
        h=e%NP
        std = 0.1
        
        m,n = xgp_W0[0].shape
        mean_ = xgp_W0[h]
        X = np.random.normal(loc=mean_, scale=std, size=(m,n))
        rgp_W0[e] = xgp_W0[h].copy() + X

        m,n = xgp_W1[0].shape
        mean_ = xgp_W1[h]
        X = np.random.normal(loc=mean_, scale=std, size=(m,n))
        rgp_W1[e] = xgp_W1[h].copy() + X

        m,n = xgp_W2[0].shape
        mean_ = xgp_W2[h]
        X = np.random.normal(loc=mean_, scale=std, size=(m,n))
        rgp_W2[e] = xgp_W2[h].copy() + X

        m,n = xgp_W3[0].shape
        mean_ = xgp_W3[h]
        X = np.random.normal(loc=mean_, scale=std, size=(m,n))
        rgp_W3[e] = xgp_W3[h].copy() + X

        m,n = xgp_b0[0].shape
        mean_ = xgp_b0[h]
        X = np.random.normal(loc=mean_, scale=std, size=(m,n))
        rgp_b0[e] = xgp_b0[h].copy() + X

        m,n = xgp_b1[0].shape
        mean_ = xgp_b1[h]
        X = np.random.normal(loc=mean_, scale=std, size=(m,n))
        rgp_b1[e] = xgp_b1[h].copy() + X

        m,n = xgp_b2[0].shape
        mean_ = xgp_b2[h]
        X = np.random.normal(loc=mean_, scale=std, size=(m,n))
        rgp_b2[e] = xgp_b2[h].copy() + X

        m,n = xgp_b3[0].shape
        mean_ = xgp_b3[h]
        X = np.random.normal(loc=mean_, scale=std, size=(m,n))
        rgp_b3[e] = xgp_b3[h].copy() + X

    local = rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3

    return local

def svd_space(M, alpha):

    # M = weight matrix
    # alpha = percent of singular values to filter - unused!!
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

def create_cluster(i, xgp, clustering_type, num_of_clusters, key):

    # reshaping for sklearn    

    # flatten each matrix
    
    d = len(xgp.keys())
    a,b = xgp[0].shape
    c = a*b
    X = np.zeros((c,d))

    for j in xgp.keys():
        X[:,j] = xgp[j].flatten()

    # clustering methods

    if clustering_type == 'kmeans':

        kmeans = KMeans(n_clusters=num_of_clusters, n_init=10, random_state=42)
        kmeans.fit(X.T)   
        c_kmeans = kmeans.predict(X.T)
        centers = kmeans.cluster_centers_
        centers = centers.T
        clabels = c_kmeans

    if clustering_type == 'spectral':
        sc = SpectralClustering(n_clusters=num_of_clusters, n_init=10, affinity='nearest_neighbors', 
                                n_neighbors = 4, random_state=42).fit(X.T)

        # determine centers from clustering

        df = pd.DataFrame.from_dict({
                'id': list(sc.labels_) ,
                'data': list(X.T)
            })
        #centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).median().agg(np.array,1) original
        centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).mean().agg(np.array,1)
        centers = centers.reset_index(drop = True)
        centers = np.array([np.broadcast_to(row, shape=(c)) for row in centers])
        centers = centers.T
        clabels = sc.labels_

    if clustering_type == 'agg':
        agg = AgglomerativeClustering(n_clusters=num_of_clusters)
        c_means = agg.fit_predict(X.T)
        clf = NearestCentroid()
        clf.fit(X.T, c_means)
        centers = clf.centroids_
        clabels = c_means
        num_of_clusters = len(centers)
        centers = centers.T

    # put centers into a 3d array

    h = np.zeros((num_of_clusters,a,b))
    k = 0
    for j in np.arange(0,num_of_clusters):
        tst = centers[:,j]
        h[k,:a,:b] = tst.reshape((a,b))
        k = k + 1

    return h

def cluster_array(xgp, clustering_type, num_of_clusters):

    # reshaping for sklearn
    # flatten each matrix
    
    d = len(xgp.keys())
    a,b = xgp[0].shape
    c = a*b
    X = np.zeros((c,d))

    for j in xgp.keys():
        X[:,j] = xgp[j].flatten()

    X = X.T

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
                                n_neighbors = d, random_state=42).fit(X)

        # determine centers from clustering

        df = pd.DataFrame.from_dict({
                'id': list(sc.labels_) ,
                'data': list(X)
            })
        #centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).median().agg(np.array,1) original
        centers = pd.DataFrame(df['data'].tolist(),index=df['id'] ).groupby(level=0).mean().agg(np.array,1)
        centers = centers.reset_index(drop = True)
        #centers = np.array([np.broadcast_to(row, shape=(d)) for row in centers])
        centers = np.array([np.broadcast_to(row, shape=(a*b)) for row in centers])
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
    center_dict = {}
        
    for j in np.arange(0,num_of_clusters):
        #center_dict[j] = centers[:,j]
        center_dict[j] = centers[:,j].reshape(a,b)

    return center_dict

def create_bootstrap_samples(x_data, y_data, num_samples, ratio_):

    x_dict = {}
    y_dict = {}

    if False: 
        l = len(x_data)
        x_ = pd.DataFrame(x_data)
        y_ = pd.DataFrame(y_data)

        for k in np.arange(0,num_samples):
            i_ = np.random.choice(l, size=int(l*ratio_), replace=True)
            xdf = x_.iloc[ list(i_),:]
            ydf = y_.iloc[ list(i_),:]

            x_dict[k] = xdf
            y_dict[k] = ydf

    # k-fold?

    # Split into 3 disjoint subsets
    #x_splits = np.array_split(x_train, 3)
    #y_splits = np.array_split(y_train, 3)

    x_disjoint = np.array_split(x_data, num_samples)
    y_disjoint = np.array_split(y_data, num_samples)

    for k in np.arange(0,num_samples):

        x_dict[k] = x_disjoint[k]
        y_dict[k] = y_disjoint[k]
    
    return x_dict, y_dict


def model_selection(gen_points, y_,return_method, c,NN_model, errors):
                   # NP_indices,scaler,optimum_point,X_,error_metric, reg_flag):
    NN_model.set_phase('testing')
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    n_ = len(y_)

    # # scaling
    # scaler.fit(X_)
    # X_ = scaler.transform(X_)

    # bumping should be evaluated on the original training dataset

    if return_method == 'bumping':
        
        final = np.argpartition(errors,range(c))[:c]
        final = final[:c]
        key_list = list(final)
        m = len(key_list)
        logging.info(f'key_list length is {m }')

        selected_point = xgp_W0[c], xgp_W1[c], xgp_W2[c], xgp_W3[c], xgp_b0[c], xgp_b1[c], xgp_b2[c], xgp_b3[c]

    if return_method in ['bma']:
        final = np.argpartition(errors,range(c))[:c]
        final = final[:c]
        key_list = list(final)
        m = len(key_list)
        logging.info(f'key_list length is {m }')

        den = sum(1/errors[key] for key in key_list)
    
        agp_W0 = sum(xgp_W0[key]*(1/errors[key]) for key in key_list)/den
        agp_W1 = sum(xgp_W1[key]*(1/errors[key]) for key in key_list)/den
        agp_W2 = sum(xgp_W2[key]*(1/errors[key]) for key in key_list)/den
        agp_W3 = sum(xgp_W3[key]*(1/errors[key]) for key in key_list)/den

        agp_b0 = sum(xgp_b0[key]*(1/errors[key]) for key in key_list)/den
        agp_b1 = sum(xgp_b1[key]*(1/errors[key]) for key in key_list)/den
        agp_b2 = sum(xgp_b2[key]*(1/errors[key]) for key in key_list)/den
        agp_b3 = sum(xgp_b3[key]*(1/errors[key]) for key in key_list)/den

        #agp_b3 = sum(xgp_b3[key]*(1/errors[key]) for key in key_list)/sum(1/errors[key] for key in key_list)

        selected_point = agp_W0, agp_W1, agp_W2, agp_W3, agp_b0, agp_b1, agp_b2, agp_b3

    return selected_point


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
        mutation_W0 = mutation_default

    if flag == 'variable':
        mutation_W0 = random.choice(mutation_list)

    if flag == 'weight_variable':
        mutation_W0 = random.choice(mutation_list)
    
    return mutation_W0


def return_mutation_list(NP):

    if NP >= 4:
        mutation_list = ['best', 'random']

    if NP >= 6:
        mutation_list = ['best', 'best2', 'random', 'random2',]

    if NP >= 8:
        mutation_list = ['best', 'best2', 'best3', 'random', 'random2', 'random3']

    return mutation_list

def candidate_measure(M, N):

    # distance matrix

    D = distance_matrix(M,N) 

    # Similarity matrix is the opposite concept to the distance matrix 

    # correlation matrix often may be considered as as a similarity matrix of variables

    D2 = np.corrcoef(M, N)

    # SVD to compare left singular vectors?

    U, S_M, V_T = svd(M)
    U, S_N, V_T = svd(N)

def plot(x_train, target, y_t, y_p, label, ext):
    plt.figure(figsize=(16,6))
    plt.plot(x_train, y_t, label='Historical', linewidth=1)
    plt.plot(x_train, y_p, color='m', label=label, linewidth=1)
    plt.xlabel('datetime')
    plt.ylabel(target)
    if target=='LMP':
        plt.ylim(-20,200)
    plt.legend(fontsize = 10, loc='upper left')
    plt.savefig(f'/home/wesley/repos/images/{ext}.png',
                format='png',
                dpi=300,
                bbox_inches='tight')    
    # plt.savefig(f'/home/wesley/repos/images/{label}.png',
    #             format='png',
    #             dpi=300,
    #             bbox_inches='tight')
    #plt.show()
    plt.close()
    #output_loc = f'/home/wesley/repos/images/houston-lmp-denn-v-test-{season}-{run}-{aindex}.csv'
    #y_2023_.to_csv(output_loc)


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

# def mutation(NP, NP_indices, F_one, F_two, F_three, x_weight, MCMC, gen_best_x_weight, mutation_type, key):

#     F_1, F_2, F_3 = return_F(key, F_one, F_two, F_three)

#     if mutation_type == 'random':

#         y = mutate(NP, NP_indices, F_1, x_weight, MCMC)

#     # DE/rand/2 needs minimum NP = 6

#     if mutation_type == 'random2':

#         y = mutate_two(NP, NP_indices, F_1, F_2, x_weight, MCMC)

#     # DE/rand/3 needs minimum NP = 8

#     if mutation_type == 'random3':

#         y = mutate_three(NP, NP_indices, F_1, F_2, F_3, x_weight, MCMC)
    
#     # DE/best/123

#     if mutation_type in ['best']:

#         y = mutate_best(NP, NP_indices, F_1, gen_best_x_weight, x_weight, MCMC)

#     if mutation_type in ['best2']:

#         y = mutate_best_two(NP, NP_indices, F_1, F_2, gen_best_x_weight, x_weight, MCMC)

#     if mutation_type in ['best3']:

#         y = mutate_best_three(NP, NP_indices, F_1, F_2, F_3, gen_best_x_weight, x_weight, MCMC)

#     return y

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


def generate_noise(b,c,MCMC):

    run_mcmc = MCMC.run_mcmc
    error_dist = MCMC.error_dist
    error_std = MCMC.error_std

    if run_mcmc and error_dist == 'norm':
        w = np.random.normal(loc=0, scale=error_std, size=(b,c))

    if run_mcmc and error_dist == 'unif':
        w = np.random.uniform(low= -error_std, high= error_std , size=(b,c))

    if not run_mcmc:
        w = np.zeros((b,c))
    return w


def crossover_component(NP_indices, y, x, CR, key):

    # crossover here is component-wise
    # at least one component per row swapped based on random int k

    z = x.copy()

    if key in ['W0','W1','W2',]:
        for e in NP_indices:
            x_ = x[e].copy()
            y_ = y[e].copy()
            z_ = z[e].copy()

            m,n = x_.shape

            for i in np.arange(0,m):
                k = np.random.choice(np.arange(0,n),) # think this should be n
                for j in np.arange(0,n):
                    if (random.uniform(0, 1) <= CR or j == k): # think this should be j
                        z_[i,j] = y_[i,j].copy()
                    else:
                        z_[i,j] = x_[i,j].copy()
            z[e] = z_.copy()

    if key in ['W3', 'b0','b1','b2', 'b3']:
        for e in NP_indices:
            x_ = x[e].copy()
            y_ = y[e].copy()
            z_ = z[e].copy()

            m,n = x_.shape

            for i in np.arange(0,m):
                k = np.random.choice(np.arange(0,n),) # think this should be n
                for j in np.arange(0,n):
                    if (random.uniform(0, 1) <= CR or j == k): # think this should be j
                        z_[i,j] = y_[i,j].copy()
                    else:
                        z_[i,j] = x_[i,j].copy()
            z[e] = z_.copy()
    return z


def crossover_vector(NP_indices, y, x, CR):

    # crossover here is column vector wise, not component-wise except for W3 and b3
    # at least one column vector swapped based on random int k

    z = x.copy()

    for e in NP_indices:
        x_ = x[e].copy()
        y_ = y[e].copy()
        z_ = z[e].copy()

        m = x_.shape
        
        for i in np.arange(0,m[0]):
            k = np.random.choice(np.arange(0,m[0]),)
            if (random.uniform(0, 1) <= CR[i] or i == k):
                z_[i] = y_[i]
            else:
                z_[i] = x_[i]
            z[e] = z_.copy()

    return z

def return_mutation_current(NP, mutation_type):

    if mutation_type in ['random', 'best']:
        k = 3
    if mutation_type in ['random2', 'best2']:
        k = 5
    if mutation_type in ['random3', 'best3']:
        k = 7
        
    permutations = math.perm(NP-1, k)

    return permutations


def mutate_perm(diffs, mutation_indices, F, x, MCMC):

    # mcmc noise addition

    b,c = x[0].shape

    y = x.copy()

    for e in mutation_indices:
        i = diffs[e][1]
        j = diffs[e][2]
        k = diffs[e][3]
        base = x[i].copy() # fix?
        v1 = x[j].copy() # fix?
        v2 = x[k].copy() # fix?
        p = base + F*(v2-v1) + generate_noise(b,c,MCMC)
        y[e] = p

    return y

def mutate_two_perm(diffs, mutation_indices, F, F2, x, MCMC):

    # mcmc noise addition

    b,c = x[0].shape

    y = x.copy()

    for e in mutation_indices:
        i = diffs[e][1]
        j = diffs[e][2]
        k = diffs[e][3]
        l = diffs[e][4]
        m = diffs[e][5]
        base = x[i]
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]
        p = base + F*(v2-v1) + F2*(v4-v3) + generate_noise(b,c,MCMC)
        y[e] = p

    return y


def mutate_three_perm(diffs, mutation_indices, F, F2, F3, x, MCMC):

    # mcmc noise addition

    b,c = x[0].shape

    y = x.copy()

    for e in mutation_indices:
        i = diffs[e][1]
        j = diffs[e][2]
        k = diffs[e][3]
        l = diffs[e][4]
        m = diffs[e][5]
        n = diffs[e][6]
        o = diffs[e][7]
        base = x[i]
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]
        v5 = x[n]
        v6 = x[o]
        p = base + F*(v2-v1) + F2*(v4-v3) + F3*(v5-v6) + generate_noise(b,c,MCMC)
        y[e] = p

    return y

def mutate_best_perm(diffs, mutation_indices, F, x, MCMC, gen_best):

    # mcmc noise addition

    b,c = x[0].shape

    y = x.copy()

    for e in mutation_indices:
        i = diffs[e][0]
        j = diffs[e][1]
        k = diffs[e][2]
        base = gen_best.copy() # fix?
        v1 = x[j].copy() # fix?
        v2 = x[k].copy() # fix?
        p = base + F*(v2-v1) + generate_noise(b,c,MCMC)
        y[e] = p

    return y


def mutate_best_two_perm(diffs, mutation_indices, F, F2, x, MCMC, gen_best):

    # mcmc noise addition

    b,c = x[0].shape

    y = x.copy()

    for e in mutation_indices:
        j = diffs[e][0]
        k = diffs[e][1]
        l = diffs[e][2]
        m = diffs[e][3]
        base = gen_best
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]
        p = base + F*(v2-v1) + F2*(v4-v3) + generate_noise(b,c,MCMC)
        y[e] = p

    return y

def mutate_best_three_perm(diffs, mutation_indices, F, F2, F3, x, MCMC, gen_best):
    
    # mcmc noise addition

    b,c = x[0].shape

    y = x.copy()

    for e in mutation_indices:
        i = diffs[e][0]
        j = diffs[e][1]
        k = diffs[e][2]
        l = diffs[e][3]
        m = diffs[e][4]
        n = diffs[e][5]
        o = diffs[e][6]
        base = gen_best.copy()
        v1 = x[j]
        v2 = x[k]
        v3 = x[l]
        v4 = x[m]
        v5 = x[n]
        v6 = x[o]
        p = base + F*(v2-v1) + F2*(v4-v3) + F3*(v5-v6) + generate_noise(b,c,MCMC)
        y[e] = p

    return y

def mutation_perm(diffs, mutation_indices, F_one, F_two, F_three, x_weight, MCMC, gen_best_x_weight, mutation_type, key):

    F_1, F_2, F_3 = return_F(key, F_one, F_two, F_three)

    if mutation_type == 'random_perm':

        y = mutate_perm(diffs, mutation_indices, F_1, x_weight, MCMC)

    # DE/rand/2 needs minimum NP = 6

    if mutation_type == 'random2_perm':

        y = mutate_two_perm(diffs, mutation_indices, F_1, F_2, x_weight, MCMC)

    # # DE/rand/3 needs minimum NP = 8

    if mutation_type == 'random3_perm':

        y = mutate_three_perm(diffs, mutation_indices, F_1, F_2, F_3, x_weight, MCMC)
    
    # DE/best/123

    if mutation_type in ['best_perm']:

        y = mutate_best_perm(diffs, mutation_indices, F_1, x_weight, MCMC, gen_best_x_weight)

    if mutation_type in ['best2_perm']:

        y = mutate_best_two_perm(diffs, mutation_indices, F_1, F_2, x_weight, MCMC, gen_best_x_weight)

    if mutation_type in ['best3_perm']:

        y = mutate_best_three_perm(diffs, mutation_indices, F_1, F_2, F_3, x_weight, MCMC, gen_best_x_weight)

    return y


def exhaustive_mutation(run, i, mutation_list, mutation_type, NP_indices, X_train, y_train, NP, DE_model,
                        F_one, F_two, F_three, pop_point, MCMC, gen_best_point, mutations_,):
    
    run_exh, exh_current_start, exh_subset = DE_model.exhaustive

    gen_best_x_W0, gen_best_x_W1, gen_best_x_W2, gen_best_x_W3, gen_best_x_b0, gen_best_x_b1, gen_best_x_b2, gen_best_x_b3 = gen_best_point
    x_W0, x_W1, x_W2, x_W3, x_b0, x_b1, x_b2, x_b3 = pop_point
    mutation_W0, mutation_W1, mutation_W2, mutation_W3, mutation_b0, mutation_b1, mutation_b2, mutation_b3 = mutations_

    #logging.info(f'gen {i} run {run}  exhaustive mutation {mutation_W0}')
    #mutation_W0, mutation_W1, mutation_W2, mutation_W3, mutation_b0, mutation_b1, mutation_b2, mutation_b3 = return_mutation_type('default', mutation_list, mutation_W0 + '_perm')
    mutation_W0, mutation_W1, mutation_W2, mutation_W3, mutation_b0, mutation_b1, mutation_b2, mutation_b3 = return_mutation_type('default', mutation_list, 'random_perm')

    # computationally prohibitive for all indices

    #indices = list(np.arange(0,NP))

    # pass in argument

    indices = list(np.arange(0,min(exh_subset,NP)))
    diffs = list(permutations(indices))
    mutation_indices = list(np.arange(0,len(diffs))) # reduce here?

    if mutation_type == 'best_perm':
        bdiffs = list(permutations(indices,3))
        mutation_indices = list(np.arange(0,len(bdiffs)))

    m_W0 = mutation_perm(diffs, mutation_indices, F_one, F_two, F_three, x_W0, MCMC, gen_best_x_W0, mutation_W0, 'W0')
    m_W1 = mutation_perm(diffs, mutation_indices, F_one, F_two, F_three, x_W1, MCMC, gen_best_x_W1, mutation_W1, 'W1')
    m_W2 = mutation_perm(diffs, mutation_indices, F_one, F_two, F_three, x_W2, MCMC, gen_best_x_W2, mutation_W2, 'W2')
    m_W3 = mutation_perm(diffs, mutation_indices, F_one, F_two, F_three, x_W3, MCMC, gen_best_x_W3, mutation_W3, 'W3')

    m_b0 = mutation_perm(diffs, mutation_indices, F_one, F_two, F_three, x_b0, MCMC, gen_best_x_b0, mutation_b0, 'b0')
    m_b1 = mutation_perm(diffs, mutation_indices, F_one, F_two, F_three, x_b1, MCMC, gen_best_x_b1, mutation_b1, 'b1')
    m_b2 = mutation_perm(diffs, mutation_indices, F_one, F_two, F_three, x_b2, MCMC, gen_best_x_b2, mutation_b2, 'b2')
    m_b3 = mutation_perm(diffs, mutation_indices, F_one, F_two, F_three, x_b3, MCMC, gen_best_x_b3, mutation_b3, 'b3')

    em_weights = m_W0, m_W1, m_W2, m_W3, m_b0, m_b1, m_b2, m_b3

    return em_weights

def skunk_initial(a, b, NP_indices, key, DE_model):
    
    std = np.sqrt(2.0 / a) # np.sqrt(2.0 / a)
    mean_ = 0 # 0

    candidates = {}

    for j in NP_indices:
        x = np.zeros((a*b,1))
        #x[j%a*b] = np.random.uniform(low=l, high=h)
        x[j%a*b] = np.random.normal(loc=mean_, scale=std)
        candidates[j] = x.reshape((a,b))

    return candidates

def selection_vector(NP_indices, fitness, error_metric_dict, X_train, y_train, gen, mindex,
              reg_flag, error_metric_, error_weight, m, n1, n2, n3,
              x, z):
    
    # determine survival of target or trial vector
    # into the next generation
    i_accept = 0
    n_ = len(y_train)

    for j in NP_indices:

        if j == mindex:
            error_metric = error_metric_
            #logging.info(f'gen {gen} is this working? {error_metric} {mindex}')
        else:
            error_metric = error_metric_dict[j]
        xcandidates = split_candidate(x[j], m, n1, n2, n3)
        xW0, xW1, xW2, xW3, xB0, xB1, xB2, xB3 = xcandidates

        zcandidates = split_candidate(z[j], m, n1, n2, n3)
        zW0, zW1, zW2, zW3, zB0, zB1, zB2, zB3 = zcandidates
        
        zfit, zyb = fitness(X_train, zW0, zW1, zW2, zW3, zB0, zB1, zB2, zB3, y_train, n_, error_metric, reg_flag, error_weight)
        xfit, xyb = fitness(X_train, xW0, xW1, xW2, xW3, xB0, xB1, xB2, xB3, y_train, n_, error_metric, reg_flag, error_weight)

        if zfit <= xfit:
            
            x[j] = z[j].copy()
            i_accept = i_accept + 1

    return x, i_accept

def construct_svd_basis(xgp_W0, mindex, maindex, NP_indices):

    sgp_W0 = {}

    for j in NP_indices:
        A = xgp_W0[mindex]
        B = xgp_W0[maindex]
        C = A+B

        U0, S0, V_T0 = svd(A)
        U1, S1, V_T1 = svd(B)
        U2, S2, V_T2 = svd(C)

        w0 = len(S2)       
        S3 = np.zeros((C.shape[0], C.shape[1]))
        S_ = np.diag(S0+S1)
        S3[:w0, :w0] = S_    

        # maybe use a different sv basis?

        D = U2 @ S3 @ V_T2
        sgp_W0[j] = D

    return D

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

def svd_exploration(NP_indices, current, i_accept, doh, refine_mod,
                       y_train, n_, error_metric, reg_flag, error_weight,
                       fitness, X_train, g_points):
    
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = g_points
    
    dgp_W0, dgp_W1, dgp_W2 = {}, {}, {}
    d_errors = []
     
    for k in NP_indices:
        dgp_W0[k] = svd_space(xgp_W0[k],doh)
        dgp_W1[k] = svd_space(xgp_W1[k],doh)
        dgp_W2[k] = svd_space(xgp_W2[k],doh)
        
    # fitness
    
    for k in NP_indices:

        d_rmse, sv = fitness(X_train, dgp_W0[k], dgp_W1[k], dgp_W2[k], xgp_W3[k], xgp_b0[k], xgp_b1[k], xgp_b2[k], xgp_b3[k],
                            y_train, n_, error_metric, reg_flag, error_weight)
        d_errors.append(d_rmse)

    # find best fitness

    d_min_value = np.amin(d_errors)
    d_index = np.where(d_errors == d_min_value)
    d_index = d_index[0][0]
    
    best_point = dgp_W0[d_index], dgp_W1[d_index], dgp_W2[d_index], xgp_W3[d_index], xgp_b0[d_index], xgp_b1[d_index], xgp_b2[d_index], xgp_b3[d_index]
        
    return d_min_value, best_point

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
    #df.loc[(df['s_log_value'] > 0) & (df['s_log_value'] <= df['TrainMin']), 'log_SVD_Count'] = 1

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

def return_bumping(return_method, dfs, MCMC, x_2023, NN_model, test_data,
                    Data, error_metric,models, application, daytype, print_master,
                    gen_points,X_train,y_train, NP_indices, reg_flag, DE_model):
                        
    logging.info(f'starting {return_method}')

    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    top = [1]
    bump_errors = []
    n_ = len(y_train)
    data = dfs[dfs['Exit'] == 'True'].copy()
    c = 1

    for j in NP_indices:
        rmse, yp = fitness(X_train, xgp_W0[j], xgp_W1[j], xgp_W2[j], xgp_W3[j], xgp_b0[j], xgp_b1[j], xgp_b2[j], xgp_b3[j], 
                        y_train, n_, error_metric, reg_flag, NN_model)
        bump_errors.append(rmse)
    
    # choose based on fitness
    
    selected_point = model_selection(gen_points, y_train,return_method, c,NN_model, bump_errors )
    gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3 = selected_point

    y_2023_pred = DE_model.DENN_forecast(x_2023, gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3, NN_model, MCMC)
    denn_rmse_2023 = root_mean_squared_error(test_data[Data.target], y_2023_pred)
    weights=None
    denn_2023_score = return_error_metric(test_data[Data.target], y_2023_pred, error_metric, weights)
    data['c'] = c
    data[f'2023_{error_metric}'] = denn_2023_score
    data['2023_RMSE'] = denn_rmse_2023
    data['TestStd'] = np.std(y_2023_pred,axis=0)[0]
    models.append(data)

    if print_master:
        xcol = 'datetime'
        label = f'DE-NN Predicted-{return_method}-{error_metric}-{application}'
        file_ext = f'houston-{application}-{daytype}-denn-test-{DE_model.run}.png'
        DE_model.plot(test_data[xcol], Data.target, test_data[Data.target], y_2023_pred, label, file_ext)

    return models, data

def return_bagging(return_method, dfs, optimum_point, x_2023, NN_model, test_data,
                    Data, error_metric,models_, application, daytype, print_master,
                    gen_points,X_train,y_train, NP_indices, reg_flag, NP, MCMC, DE_model):
    top = list(np.arange(1,NP+1))
    model_rmse = []
    #models_ = []
    a = len(x_2023)
    b = 1
    T = np.zeros((NP,a,b))

    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points

    errors = []
    bag_fitness = []
    C = []
    print(NP_indices)
    #print(models)
    for j in NP_indices:
        logging.info(f'starting {return_method} {j}')
        data = dfs[dfs['Exit'] == 'True'].copy()
        yp = DE_model.DENN_forecast(x_2023, xgp_W0[j], xgp_W1[j], xgp_W2[j], xgp_W3[j], xgp_b0[j], xgp_b1[j], xgp_b2[j], xgp_b3[j], NN_model, MCMC)
        weights=None
        current_fitness = return_error_metric(test_data[Data.target], yp, error_metric, weights)
        bag_fitness.append(current_fitness)
        
        T[j,:,:] = yp
        #y_avg_p = np.mean(T,axis=0)
        y_avg_p = np.sum(T,axis=0) / (j+1)
        C.append(y_avg_p)
        
        denn_2023_score = return_error_metric(test_data[Data.target], y_avg_p, error_metric, weights)
        denn_rmse_2023 = root_mean_squared_error(test_data[Data.target], y_avg_p)
        errors.append(denn_2023_score)
        data['c'] = j
        data[f'2023_{error_metric}'] = denn_2023_score
        data['2023_RMSE'] = denn_rmse_2023
        data['TestStd'] = np.std(y_avg_p,axis=0)[0]
        models_.append(data)
        #print(models)

    # find best bagging index                        
    
    bagging_fitness = np.array(errors)
    min_value = np.amin(bagging_fitness)
    mindex = np.where(bagging_fitness == min_value)
    mindex = mindex[0][0] # index integer
    
    #y_avg_pc = np.mean(T[0:mindex,:,:],axis=0)
    y_avg_pc = C[mindex]

    if print_master:
        xcol = 'datetime'
        label = f'DE-NN Predicted-{return_method}-{error_metric}-{application}'
        file_ext = f'houston-{application}-{daytype}-denn-test-{DE_model.run}-{mindex}.png'
        DE_model.plot(test_data[xcol], Data.target, test_data[Data.target], y_avg_pc, label, file_ext)

    return models_, data

def return_bma(return_method, dfs, optimum_point, x_2023, NN_model, test_data,
                                        Data, error_metric,models, application, daytype, print_master,
                                        gen_points,X_train_scaled, y_train, NP_indices, reg_flag, NP,MCMC, DE_model):
    top = list(np.arange(1,NP+1))
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    a_W0, a_W1, a_W2, a_W3, a_b0, a_b1, a_b2, a_b3 = {}, {}, {}, {}, {}, {}, {}, {},
    model_rmse = []
    n_ = len(y_train)
    bma_errors = []
    bma_std = []
    NN_model.set_phase('testing')
    for j in NP_indices:
        #write_weights(xgp_W0[j], xgp_W1[j], xgp_W2[j], xgp_W3[j], xgp_b0[j], xgp_b1[j], xgp_b2[j], xgp_b3[j],j, return_method,X_,y_)
        fit_score, ypp = fitness(X_train_scaled, xgp_W0[j], xgp_W1[j], xgp_W2[j], xgp_W3[j], xgp_b0[j], xgp_b1[j], xgp_b2[j], xgp_b3[j], 
                        y_train, n_, error_metric, reg_flag, NN_model)
        bma_errors.append(fit_score)
        yp_std = np.std(ypp,axis=0)[0]
        bma_std.append(yp_std)

    # weight based on fitness
    
    for c in top:
        logging.info(f'starting {return_method} {c}')
        data = dfs[dfs['Exit'] == 'True'].copy()

        selected_point = model_selection(gen_points, y_train,return_method, c,NN_model, bma_errors)
        gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3 = selected_point

        #

        y_2023_pred = DE_model.DENN_forecast(x_2023, gb_W0, gb_W1, gb_W2, gb_W3,
                            gb_b0, gb_b1, gb_b2, gb_b3, NN_model, MCMC)
        denn_rmse_2023 = root_mean_squared_error(test_data[Data.target], y_2023_pred)
        weights=None
        denn_2023_score = return_error_metric(test_data[Data.target], y_2023_pred, error_metric, weights)
        #n_ = len(x_2023)
        #yp_check = feed_forward(x_2023, gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3, n_, NN_model)
        data['c'] = c
        data[f'2023_{error_metric}'] = denn_2023_score
        data['2023_RMSE'] = denn_rmse_2023
        data['TestStd'] = np.std(y_2023_pred,axis=0)[0]
        models.append(data)

        # model averaging

        model_rmse.append(denn_rmse_2023)

        a_W0[int(c)] = gb_W0
        a_W1[int(c)] = gb_W1
        a_W2[int(c)] = gb_W2
        a_W3[int(c)] = gb_W3

        a_b0[int(c)] = gb_b0
        a_b1[int(c)] = gb_b1
        a_b2[int(c)] = gb_b2
        a_b3[int(c)] = gb_b3                     
    
    # determine best model averaging candidate

    if print_master:

        model_fitness_values = np.array(model_rmse)
        min_value = np.amin(model_fitness_values)
        aindex = np.where(model_fitness_values == min_value)
        aindex = aindex[0][0] + 1 # index integer

        xcol = 'datetime'
        m_W0, m_W1, m_W2, m_W3, m_b0, m_b1, m_b2, m_b3 = a_W0[aindex], a_W1[aindex], a_W2[aindex], a_W3[aindex], a_b0[aindex], a_b1[aindex], a_b2[aindex], a_b3[aindex]
        y_2023_ = DE_model.DENN_forecast(x_2023, m_W0, m_W1, m_W2, m_W3, m_b0, m_b1, m_b2, m_b3, NN_model, MCMC)
        label = f'DE-NN Predicted-{return_method}-{error_metric}-{application}'
        file_ext = f'houston-{application}-{daytype}-denn-test-{DE_model.run}-{aindex}.png'
        plot(test_data[xcol], Data.target, test_data[Data.target], y_2023_, label, file_ext)

    return models, data

def return_bma_val(return_method, dfs, optimum_point, x_2023, NN_model, test_data,
                                        Data, error_metric,models, application, daytype, print_master,
                                        val_points,X_train_scaled, y_train, NP_indices, reg_flag, NP,MCMC, DE_model):
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = val_points
    a_W0, a_W1, a_W2, a_W3, a_b0, a_b1, a_b2, a_b3 = {}, {}, {}, {}, {}, {}, {}, {}
    key_list = xgp_W0.keys()
    model_rmse = []
    n_ = len(y_train)
    errors = {}
    bma_std = []
    NN_model.set_phase('testing')
    #for j in NP_indices:
    for j in key_list:
        #write_weights(xgp_W0[j], xgp_W1[j], xgp_W2[j], xgp_W3[j], xgp_b0[j], xgp_b1[j], xgp_b2[j], xgp_b3[j],j, return_method,X_,y_)
        fit_score, ypp = fitness(X_train_scaled, xgp_W0[j], xgp_W1[j], xgp_W2[j], xgp_W3[j], xgp_b0[j], xgp_b1[j], xgp_b2[j], xgp_b3[j], 
                        y_train, n_, error_metric, reg_flag, NN_model)
        errors[j]=fit_score
        #errors.append(fit_score)
        yp_std = np.std(ypp,axis=0)[0]
        bma_std.append(yp_std)

    # weight based on fitness

    #iidx = np.argpartition(errors, DE_model.val_sample-1)[:DE_model.val_sample]
    #key_list = iidx[:DE_model.val_sample]
    
    #for c in top:
    c = len(key_list)
    logging.info(f'starting {return_method} {c}')
    data = dfs[dfs['Exit'] == 'True'].copy()

    ###
    #key_list = xgp_W0.keys()
    den = sum(1/errors[key] for key in key_list)

    agp_W0 = sum(xgp_W0[key]*(1/errors[key]) for key in key_list)/den
    agp_W1 = sum(xgp_W1[key]*(1/errors[key]) for key in key_list)/den
    agp_W2 = sum(xgp_W2[key]*(1/errors[key]) for key in key_list)/den
    agp_W3 = sum(xgp_W3[key]*(1/errors[key]) for key in key_list)/den

    agp_b0 = sum(xgp_b0[key]*(1/errors[key]) for key in key_list)/den
    agp_b1 = sum(xgp_b1[key]*(1/errors[key]) for key in key_list)/den
    agp_b2 = sum(xgp_b2[key]*(1/errors[key]) for key in key_list)/den
    agp_b3 = sum(xgp_b3[key]*(1/errors[key]) for key in key_list)/den

    #

    y_2023_pred = DE_model.DENN_forecast(x_2023, agp_W0, agp_W1, agp_W2, agp_W3,
                        agp_b0, agp_b1, agp_b2, agp_b3, NN_model, MCMC)
    
    denn_rmse_2023 = root_mean_squared_error(test_data[Data.target], y_2023_pred)
    weights=None
    denn_2023_score = return_error_metric(test_data[Data.target], y_2023_pred, error_metric, weights)
    #n_ = len(x_2023)
    #yp_check = feed_forward(x_2023, gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3, n_, NN_model)
    data['c'] = c
    data[f'2023_{error_metric}'] = denn_2023_score
    data['2023_RMSE'] = denn_rmse_2023
    data['TestStd'] = np.std(y_2023_pred,axis=0)[0]
    models.append(data)

    # model averaging

    model_rmse.append(denn_rmse_2023)                  
    
    # determine best model averaging candidate

    if print_master:

        model_fitness_values = np.array(model_rmse)
        min_value = np.amin(model_fitness_values)
        aindex = np.where(model_fitness_values == min_value)
        aindex = aindex[0][0] + 1 # index integer

        xcol = 'datetime'
        m_W0, m_W1, m_W2, m_W3, m_b0, m_b1, m_b2, m_b3 =  agp_W0, agp_W1, agp_W2, agp_W3, agp_b0, agp_b1, agp_b2, agp_b3
        y_2023_ = DE_model.DENN_forecast(x_2023, m_W0, m_W1, m_W2, m_W3, m_b0, m_b1, m_b2, m_b3, NN_model, MCMC)
        label = f'DE-NN Predicted-{return_method}-{error_metric}-{application}'
        file_ext = f'houston-{application}-{daytype}-denn-test-{DE_model.run}-{aindex}.png'
        plot(test_data[xcol], Data.target, test_data[Data.target], y_2023_, label, file_ext)

    return models, data



def perform_clustering(NP,bootstrapping, X_train, y_train, min_value, maindex, DE_model,
                       reg_flag, NN_model, n_, gen_points, i):
    
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    clustering_list = ['kmeans', 'spectral', 'agg']
    clustering_type = random.choice(clustering_list)

    num_of_clusters_list = list(np.arange(2,NP-2))
    if NP == 4:
        num_of_clusters_list = [3]
    num_of_clusters = random.choice(num_of_clusters_list)
    
    cgp_W0 = cluster_array(xgp_W0, clustering_type, num_of_clusters)
    cgp_W1 = cluster_array(xgp_W1, clustering_type, num_of_clusters)
    cgp_W2 = cluster_array(xgp_W2, clustering_type, num_of_clusters)
    cgp_W3 = cluster_array(xgp_W3, clustering_type, num_of_clusters)

    cgp_b0 = cluster_array(xgp_b0, clustering_type, num_of_clusters)
    cgp_b1 = cluster_array(xgp_b1, clustering_type, num_of_clusters)
    cgp_b2 = cluster_array(xgp_b2, clustering_type, num_of_clusters)
    cgp_b3 = cluster_array(xgp_b3, clustering_type, num_of_clusters)

    # fitness

    c_errors = []

    for s in np.arange(0,num_of_clusters):
        if bootstrapping:
            X_train_ = X_train[s%NP]
            y_train_ = y_train[s%NP]
            m_ = len(y_train_)
            c_fit, cv = fitness(X_train_, cgp_W0[s], cgp_W1[s], cgp_W2[s], cgp_W3[s], cgp_b0[s], cgp_b1[s], cgp_b2[s], cgp_b3[s],
                            y_train_, m_, DE_model.error_metric, reg_flag, NN_model)
        else:
            c_fit, cv = fitness(X_train, cgp_W0[s], cgp_W1[s], cgp_W2[s], cgp_W3[s], cgp_b0[s], cgp_b1[s], cgp_b2[s], cgp_b3[s],
                            y_train, n_, DE_model.error_metric, reg_flag, NN_model)
        c_errors.append(c_fit)

    # find best fitness

    c_min_value = np.amin(c_errors)
    c_index = np.where(c_errors == c_min_value)
    c_index = c_index[0][0]

    if c_min_value < min_value:
        #logging.info(f'gen {i} {clustering_type} {num_of_clusters} clustering min {c_min_value} max {max_value}')
        logging.info(f'gen {i} {clustering_type} {num_of_clusters} clustering min {c_min_value} min {min_value}')
        xgp_W0[maindex] = cgp_W0[c_index].copy()
        xgp_W1[maindex] = cgp_W1[c_index].copy()
        xgp_W2[maindex] = cgp_W2[c_index].copy()
        xgp_W3[maindex] = cgp_W3[c_index].copy()

        xgp_b0[maindex] = cgp_b0[c_index].copy()
        xgp_b1[maindex] = cgp_b1[c_index].copy()
        xgp_b2[maindex] = cgp_b2[c_index].copy()
        xgp_b3[maindex] = cgp_b3[c_index].copy()
    
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    return gen_points, c_min_value

def perform_search(NP,bootstrapping, X_train, y_train, min_value, maindex, DE_model,
                        reg_flag, NN_model, n_, gen_points,i, NP_indices, current):
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    local_ = 20
    samples = local_ * (int(current/1000) + 1)
    
    #logging.INFO(f'gen {i} STARTING uniform local search samples {samples}')
    #logging.INFO(f'gen {i} STARTING uniform local search samples')
    local = random_uniform(xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3, samples, NP_indices)
    rgp_W0, rgp_W1, rgp_W2, rgp_W3, rgp_b0, rgp_b1, rgp_b2, rgp_b3 = local
    
    # fitness

    l_errors = []
    l = samples*NP

    for s in np.arange(0,l):
        if bootstrapping:
            X_train_ = X_train[s%NP]
            y_train_ = y_train[s%NP]
            m_ = len(y_train_)
            l_fit, rv = fitness(X_train_, rgp_W0[s], rgp_W1[s], rgp_W2[s], rgp_W3[s], rgp_b0[s], rgp_b1[s], rgp_b2[s], rgp_b3[s],
                                y_train_, m_, DE_model.error_metric, reg_flag, NN_model)
        else:
            l_fit, yb = fitness(X_train, rgp_W0[s], rgp_W1[s], rgp_W2[s], rgp_W3[s], rgp_b0[s], rgp_b1[s], rgp_b2[s], rgp_b3[s],
                    y_train, n_, DE_model.error_metric, reg_flag, NN_model) # gen_train_score
        l_errors.append(l_fit)
        
        if l_fit < min_value:
            logging.info(f'gen {i} uniform local search {l_fit} min {min_value} samples {samples}')    
            xgp_W0[s%NP] = rgp_W0[s].copy()
            xgp_W1[s%NP] = rgp_W1[s].copy()
            xgp_W2[s%NP] = rgp_W2[s].copy()
            xgp_W3[s%NP] = rgp_W3[s].copy()

            xgp_b0[s%NP] = rgp_b0[s].copy()
            xgp_b1[s%NP] = rgp_b1[s].copy()
            xgp_b2[s%NP] = rgp_b2[s].copy()
            xgp_b3[s%NP] = rgp_b3[s].copy()
            break
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    return gen_points, l_fit 


def perform_svd_filter(NP,bootstrapping, X_train, y_train, min_value, maindex, DE_model,
                        reg_flag, NN_model, n_, gen_points,i, NP_indices, current):
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    dgp_W0, dgp_W1, dgp_W2 = {},{},{}
    test_errors = []

    S = 0
    for k in NP_indices:
        for j in [1,2]:
            dgp_W0[S] = svd_space(xgp_W0[k], j)
            dgp_W1[S] = svd_space(xgp_W1[k], j)
            dgp_W2[S] = svd_space(xgp_W2[k], j)
            S = S+1

    for s in np.arange(0,S):
        if bootstrapping:
            X_train_ = X_train[s%NP]
            y_train_ = y_train[s%NP]
            m_ = len(y_train_)
            svd_fit, dv = fitness(X_train_, dgp_W0[s], dgp_W1[s], dgp_W2[s], xgp_W3[s%NP], xgp_b0[s%NP], xgp_b1[s%NP], xgp_b2[s%NP], xgp_b3[s%NP],
                            y_train_, m_, DE_model.error_metric, reg_flag, NN_model)
        else:
            svd_fit, dv = fitness(X_train, dgp_W0[s], dgp_W1[s], dgp_W2[s], xgp_W3[s%NP], xgp_b0[s%NP], xgp_b1[s%NP], xgp_b2[s%NP], xgp_b3[s%NP],
                            y_train, n_, DE_model.error_metric, reg_flag, NN_model)
        test_errors.append(svd_fit)

        if svd_fit < min_value:
            #logging.info(f'gen {i} svd filter {svd_value} max {max_value}')
            logging.info(f'gen {i} svd filter {svd_fit} min {min_value}')
            xgp_W0[s%NP] = dgp_W0[s].copy()
            xgp_W1[s%NP] = dgp_W1[s].copy()
            xgp_W2[s%NP] = dgp_W2[s].copy()
            xgp_W3[s%NP] = xgp_W3[s%NP].copy()

            xgp_b0[s%NP] = xgp_b0[s%NP].copy()
            xgp_b1[s%NP] = xgp_b1[s%NP].copy()
            xgp_b2[s%NP] = xgp_b2[s%NP].copy()
            xgp_b3[s%NP] = xgp_b3[s%NP].copy()
            break
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    return gen_points, svd_fit 


def perform_svd_scalar(NP,bootstrapping, X_train, y_train, min_value, maindex, DE_model,
                        reg_flag, NN_model, n_, gen_points,i, NP_indices, current):
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    dgp_W0, dgp_W1, dgp_W2 = {},{},{}
    test_errors = []

    S = 0
    for k in NP_indices:
        U_W0, S_W0, V_T_W0 = svd(xgp_W0[k])
        U_W1, S_W1, V_T_W1 = svd(xgp_W1[k])
        U_W2, S_W2, V_T_W2 = svd(xgp_W2[k])

        for j in np.arange(0,2,0.1):
            dgp_W0[S] = reconstruct_SVD(U_W0, S_W0*j, V_T_W0)
            dgp_W1[S] = reconstruct_SVD(U_W1, S_W1*j, V_T_W1)
            dgp_W2[S] = reconstruct_SVD(U_W2, S_W2*j, V_T_W2)
            S = S+1

    for s in np.arange(0,S):
        if bootstrapping:
            X_train_ = X_train[s%NP]
            y_train_ = y_train[s%NP]
            m_ = len(y_train_)
            s_scalar_value, dv = fitness(X_train_, dgp_W0[s], dgp_W1[s], dgp_W2[s], xgp_W3[s%NP], xgp_b0[s%NP], xgp_b1[s%NP], xgp_b2[s%NP], xgp_b3[s%NP],
                            y_train_, m_, DE_model.error_metric, reg_flag, NN_model)
        else:
            s_scalar_value, dv = fitness(X_train, dgp_W0[s], dgp_W1[s], dgp_W2[s], xgp_W3[s%NP], xgp_b0[s%NP], xgp_b1[s%NP], xgp_b2[s%NP], xgp_b3[s%NP],
                            y_train, n_, DE_model.error_metric, reg_flag, NN_model)
            
        test_errors.append(s_scalar_value)

        if s_scalar_value < min_value:
            #logging.info(f'gen {i} svd scalar {s_scalar_value} max {max_value}')
            logging.info(f'gen {i} svd scalar {s_scalar_value} min {min_value}')
            xgp_W0[s%NP] = dgp_W0[s].copy()
            xgp_W1[s%NP] = dgp_W1[s].copy()
            xgp_W2[s%NP] = dgp_W2[s].copy()
            xgp_W3[s%NP] = xgp_W3[s%NP].copy()

            xgp_b0[s%NP] = xgp_b0[s%NP].copy()
            xgp_b1[s%NP] = xgp_b1[s%NP].copy()
            xgp_b2[s%NP] = xgp_b2[s%NP].copy()
            xgp_b3[s%NP] = xgp_b3[s%NP].copy()
            break

    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    return gen_points, s_scalar_value 

def perform_svd_exp(NP,bootstrapping, X_train, y_train, min_value, maindex, DE_model,
                        reg_flag, NN_model, n_, gen_points,i, NP_indices, current):
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points

    dgp_W0, dgp_W1, dgp_W2 = {},{},{}
    test_errors = []

    S = 0
    for k in NP_indices:
        U_W0, S_W0, V_T_W0 = svd(xgp_W0[k])
        U_W1, S_W1, V_T_W1 = svd(xgp_W1[k])
        U_W2, S_W2, V_T_W2 = svd(xgp_W2[k])

        #for j in np.arange(1.05,1.1,0.01):
        for j in np.arange(1.01,1.2,0.01):
            dgp_W0[S] = reconstruct_SVD(U_W0, S_W0*j, V_T_W0)
            dgp_W1[S] = reconstruct_SVD(U_W1, S_W1*j, V_T_W1)
            dgp_W2[S] = reconstruct_SVD(U_W2, S_W2*j, V_T_W2)
            S = S+1

    for s in np.arange(0,S):
        if bootstrapping:
            X_train_ = X_train[s%NP]
            y_train_ = y_train[s%NP]
            m_ = len(y_train_)
            s_exp_value, dv = fitness(X_train_, dgp_W0[s], dgp_W1[s], dgp_W2[s], xgp_W3[s%NP], xgp_b0[s%NP], xgp_b1[s%NP], xgp_b2[s%NP], xgp_b3[s%NP],
                            y_train_, m_, DE_model.error_metric, reg_flag, NN_model)
        else:
            s_exp_value, dv = fitness(X_train, dgp_W0[s], dgp_W1[s], dgp_W2[s], xgp_W3[s%NP], xgp_b0[s%NP], xgp_b1[s%NP], xgp_b2[s%NP], xgp_b3[s%NP],
                            y_train, n_, DE_model.error_metric, reg_flag, NN_model)

        test_errors.append(s_exp_value)

        if s_exp_value < min_value:
            #logging.info(f'gen {i} svd exp {s_exp_value} max {max_value}')
            logging.info(f'gen {i} svd exp {s_exp_value} min {min_value}')
            xgp_W0[s%NP] = dgp_W0[s].copy()
            xgp_W1[s%NP] = dgp_W1[s].copy()
            xgp_W2[s%NP] = dgp_W2[s].copy()
            xgp_W3[s%NP] = xgp_W3[s%NP].copy()

            xgp_b0[s%NP] = xgp_b0[s%NP].copy()
            xgp_b1[s%NP] = xgp_b1[s%NP].copy()
            xgp_b2[s%NP] = xgp_b2[s%NP].copy()
            xgp_b3[s%NP] = xgp_b3[s%NP].copy()
            break

    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    return gen_points, s_exp_value 

def perform_svd_log(NP,bootstrapping, X_train, y_train, min_value, maindex, DE_model,
                        reg_flag, NN_model, n_, gen_points,i, NP_indices, current):
    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    dgp_W0, dgp_W1, dgp_W2 = {},{},{}
    test_errors = []

    S = 0
    for k in NP_indices:
        U_W0, S_W0, V_T_W0 = svd(xgp_W0[k])
        U_W1, S_W1, V_T_W1 = svd(xgp_W1[k])
        U_W2, S_W2, V_T_W2 = svd(xgp_W2[k])

        for j in np.arange(4,5.05,0.05):
            dgp_W0[S] = reconstruct_SVD(U_W0, np.log( np.diag(S_W0) + 1)**j, V_T_W0)
            dgp_W1[S] = reconstruct_SVD(U_W1, np.log( np.diag(S_W1) + 1)**j, V_T_W1)
            dgp_W2[S] = reconstruct_SVD(U_W2, np.log( np.diag(S_W2) + 1)**j, V_T_W2)
            S = S+1

    for s in np.arange(0,S):
        if bootstrapping:
            X_train_ = X_train[s%NP]
            y_train_ = y_train[s%NP]
            m_ = len(y_train_)
            s_log_value, dv = fitness(X_train_, dgp_W0[s], dgp_W1[s], dgp_W2[s], xgp_W3[s%NP], xgp_b0[s%NP], xgp_b1[s%NP], xgp_b2[s%NP], xgp_b3[s%NP],
                            y_train_, m_, DE_model.error_metric, reg_flag, NN_model)
        else:
            s_log_value, dv = fitness(X_train, dgp_W0[s], dgp_W1[s], dgp_W2[s], xgp_W3[s%NP], xgp_b0[s%NP], xgp_b1[s%NP], xgp_b2[s%NP], xgp_b3[s%NP],
                            y_train, n_, DE_model.error_metric, reg_flag, NN_model)
            
        test_errors.append(s_log_value)

        if s_log_value < min_value:
            #logging.info(f'gen {i} svd log {s_log_value} max {max_value}')
            logging.info(f'gen {i} svd log {s_log_value} min {min_value}')
            xgp_W0[s%NP] = dgp_W0[s].copy()
            xgp_W1[s%NP] = dgp_W1[s].copy()
            xgp_W2[s%NP] = dgp_W2[s].copy()
            xgp_W3[s%NP] = xgp_W3[s%NP].copy()

            xgp_b0[s%NP] = xgp_b0[s%NP].copy()
            xgp_b1[s%NP] = xgp_b1[s%NP].copy()
            xgp_b2[s%NP] = xgp_b2[s%NP].copy()
            xgp_b3[s%NP] = xgp_b3[s%NP].copy()
            break

    xgp_W0, xgp_W1, xgp_W2, xgp_W3, xgp_b0, xgp_b1, xgp_b2, xgp_b3 = gen_points
    return gen_points, s_log_value 

def post_DE(post_de_args, de_output):
    optimum_point, gen_points, val_points, dfs = de_output
    error_metric, models,DE_model, NP_indices, return_method, print_master, NP, DE_model = post_de_args
    #name = 'DE'
    #success = write_weights( W0_, W1_, W2_, W3_, b0_, b1_, b2_, b3_,run)

    # can derive multiple models from DE run, e.g some sort of model averaging

    # generate bootstrapped samples from original dataset
    # fit model to each
    # produce forecast from each bootstrapped fitted model and take the mean

    # it looks like bagging with bootstrapping=True is bagging
    # bagging without bootstrapping is not bagging

    if DE_model.return_method == 'bagging':
        models, data = return_bagging(return_method, dfs, optimum_point, x_2023, NN_model, test_data,
                                        Data, error_metric,models, application, daytype, print_master,
                                        gen_points,X_train_scaled,y_train, NP_indices, reg_flag, NP, MCMC, DE_model)

    # it looks like standard with bootstrapping=True is bumping

    if DE_model.return_method in ['standard', 'standard_val']:
        models, data = return_standard(return_method, dfs, optimum_point, error_metric,models, print_master, DE_model)
    
    # it looks like standard with bootstrapping=True is bumping
    # when return_method == 'bumping', bootstrapping=True.

    if DE_model.return_method == 'bumping':
        models, data = return_bumping(return_method, dfs, MCMC, x_2023, NN_model, test_data,
                        Data, error_metric,models, application, daytype, print_master,
                        gen_points,X_train_scaled, y_train, NP_indices, reg_flag, DE_model)
    
    # BMA

    if DE_model.return_method == 'bma':
        models, data = return_bma(return_method, dfs, optimum_point, x_2023, NN_model, test_data,
                        Data, error_metric,models, application, daytype, print_master,
                        gen_points,X_train_scaled, y_train, NP_indices, reg_flag, NP,MCMC, DE_model)

    # validation BMA
    
    if DE_model.return_method == 'bma_val':
        models, data = return_bma_val(return_method, dfs, optimum_point, x_2023, NN_model, test_data,
                        Data, error_metric,models, application, daytype, print_master,
                        val_points,X_train_scaled, y_train, NP_indices, reg_flag, NP,MCMC, DE_model)
    
    return models, data


def post_DE_MCMC(post_de_mcmc_args, args2, mcmc_chain,G):

    optimum_point, gen_points, val_points, dfs, scaler, X_train_scaled, y_train = args2
    application, daytype, num_layers, NN_model, Data, test_data, x_2023, error_metric,\
    models,ycol, DE_model, NP_indices, reg_flag, return_method, print_master, NP, MCMC, DE_model = post_de_mcmc_args

    gb_W0, gb_W1, gb_W2, gb_W3, gb_b0, gb_b1, gb_b2, gb_b3 = gen_points

    burn_in = MCMC.burn_in
    pred_post_sample = MCMC.pred_post_sample
    parallel_chain = MCMC.parallel_chain
    model_name = application + ' ' + daytype
    
    # bayesian neural network
    # single model (set of weight matrics and bias vectors) samples for forecast
    # mode of each posterior
    
    if num_layers == 3:
        W0, W1, W2, W3, b0, b1, b2, b3 = mcmc_chain

        data = dfs[dfs['Exit'] == 'True'].copy()
        W0_, W0_fit, W0_T = return_distribution_mode(gb_W0, W0, 'W0', DE_model.run, parallel_chain, MCMC, model_name)
        W1_, W1_fit, W1_T = return_distribution_mode(gb_W1, W1, 'W1', DE_model.run, parallel_chain, MCMC, model_name)
        W2_, W2_fit, W2_T = return_distribution_mode(gb_W2, W2, 'W2', DE_model.run, parallel_chain, MCMC, model_name)
        W3_, W3_fit, W3_T = return_distribution_mode(gb_W3, W3, 'W3', DE_model.run, parallel_chain, MCMC, model_name)
            
        b0_, b0_fit, b0_T = return_distribution_mode(gb_b0, b0, 'b0', DE_model.run, parallel_chain, MCMC, model_name)
        b1_, b1_fit, b1_T = return_distribution_mode(gb_b1, b1, 'b1', DE_model.run, parallel_chain, MCMC, model_name)
        b2_, b2_fit, b2_T = return_distribution_mode(gb_b2, b2, 'b2', DE_model.run, parallel_chain, MCMC, model_name)
        b3_, b3_fit, b3_T = return_distribution_mode(gb_b3, b3, 'b3', DE_model.run, parallel_chain, MCMC, model_name)

        plot_predictive = True

        # need to batch for larger chain
        # populating such large 3d arrays runs out of memory

        if pred_post_sample not in ['default']:
            total_sample_length = pred_post_sample
        else:
            total_sample_length = len(W0_T)

        print(f'total chain length {total_sample_length} pred_post_sample {pred_post_sample}')
        
        batch_size = int(min(G-burn_in,total_sample_length)) # can't exceed number of samples: produces all zeros
        multiple = int(total_sample_length/batch_size)
        num_target = len(test_data[Data.target])

        M = np.zeros((total_sample_length,num_target,1))
        
        for w in np.arange(0,multiple):
            batch_start = w*batch_size
            batch_end = (w+1)*batch_size
            samples_pred_batch = DE_model.DENN_forecast(x_2023, W0_T[batch_start:batch_end], W1_T[batch_start:batch_end], W2_T[batch_start:batch_end], W3_T[batch_start:batch_end],
                                                b0_T[batch_start:batch_end], b1_T[batch_start:batch_end], b2_T[batch_start:batch_end], b3_T[batch_start:batch_end], 
                                                NN_model, MCMC)
            M[batch_start:batch_end,:,:] = samples_pred_batch

        # mean of posterior predictive samples
        
        chickenbutt = np.mean(M[-total_sample_length:,:,:],axis=0)
        rmse_mean_pred = root_mean_squared_error(test_data[Data.target], chickenbutt)
        samples_pred = M.copy()
        logging.critical(f'posterior predictive samples {len(samples_pred)}')

        if plot_predictive:
            xcol = 'datetime'
            boo = test_data.copy()
            boo['pred_mean'] = np.mean(samples_pred, axis=0)
            boo['actual'] = test_data[Data.target]
            boo['lower'] = np.percentile(samples_pred, 5, axis=0)
            boo['upper'] = np.percentile(samples_pred, 95, axis=0)
            boohoo = pd.DataFrame(boo, columns = ['datetime', 'pred_mean', 'actual', 'lower', 'upper'])
            boohoo.index = pd.DatetimeIndex(boohoo.datetime)

            dataset = boohoo.asfreq('h')
            dataset['datetime'] = dataset.index
            dataset['Month'] = dataset['datetime'].dt.month

            plot_CI(dataset, xcol, samples_pred, application, daytype, DE_model.run, 'year')

            sum_list = [6]
            summer_mask = dataset['Month'].isin(sum_list)
            jdataset = dataset[summer_mask].copy()

            plot_CI(jdataset, xcol, samples_pred, application, daytype, DE_model.run, 'june')

            # computational approximation for weight/bias posterior
            # can use fitted distribution for predictive posterior
            # define distribution and sample from it
            # construct 3d arrays from posterior samples

            # weight/bias mode forecast

        y_2023_mode_pred = DE_model.DENN_forecast(x_2023, W0_, W1_, W2_, W3_, b0_, b1_, b2_, b3_, NN_model, MCMC)
        mode_rmse_2023 = root_mean_squared_error(test_data[Data.target], y_2023_mode_pred)
        weights=None
        pred_mode_2023_score = return_error_metric(test_data[Data.target], y_2023_mode_pred, error_metric, weights)
        data[f'2023_{error_metric}_MCMC_mode'] = pred_mode_2023_score
        data['2023_RMSE_MCMC_mode'] = mode_rmse_2023
        data['c'] = total_sample_length
        data['pred_post_sample'] = pred_post_sample
        data['TestStd'] = np.std(y_2023_mode_pred,axis=0)[0]
        data['2023_RMSE_MCMC_mean'] = rmse_mean_pred

        if parallel_chain:

            # gelman-rubin convergence diagnostic

            gb_df, diag = gelman_rubin(mcmc_chain, gen_points, DE_model.run)
            write_gelman(gb_df, diag, DE_model.run)

        models.append(data)
        xcol = 'datetime'
        label = f'DE-NN Predicted-MCMC-{error_metric}-{application}'
        file_ext = f'{application}-{daytype}-denn-test-{DE_model.run}.png'
        DE_model.plot(test_data[xcol], ycol, test_data[Data.target], y_2023_mode_pred, label, file_ext)
    name='mcmc'
    #success = write_weights( W0_, W1_, W2_, W3_, b0_, b1_, b2_, b3_,run,name)

    return models, data

def return_layers(application,daytype,MCMC):
    if application == 'lmp':
        layers = (5,5,1)
    if daytype == 'weekday':
        #layers = (5,5,4)
        #layers = (4,3,4)
        layers = (150,100,50)
    if daytype == 'weekend':
        layers = (4,4,3)

    if MCMC.run_mcmc:
        if application == 'lmp':
            layers = (5,5,1)
        if daytype == 'weekday':
            #layers = (4,3,4)
            layers = (100,75,50)
        if daytype == 'weekend':
            layers = (4,4,3)

    return layers

@ray.remote
def process(other_args, de_input, run):
    application,daytype,num_layers,test_data,error_metric,ycol,reg_flag, return_method, print_master, NP, models = other_args

    DE_model, NN_model, Data, MCMC, train_size = de_input
    #DE_model.set_run(os.getpid())
    DE_model.set_run(run)
    optimum_point, gen_points, val_points, dfs, mcmc_chain, scaler, X_train_scaled, y_train = differential_evolution(DE_model, NN_model, Data, MCMC, train_size)
    de_output = optimum_point, gen_points, val_points, dfs, scaler, X_train_scaled, y_train 

    # model selection

    NP_indices = list(np.arange(0,NP))
    #X_ = x_2020
    #y_ = training_data[Data.target]

    # 2023

    x_2023 = test_data[Data.x_cols].copy()
    x_2023 = scaler.transform(x_2023)

    #models=[]    

    post_de_args = application, daytype, num_layers, NN_model, Data, test_data, x_2023, error_metric,\
                                models,ycol, DE_model, NP_indices, reg_flag, return_method, print_master, NP, MCMC, DE_model

    if not MCMC.run_mcmc:
        models, data = post_DE(post_de_args, de_output)
        
    if MCMC.run_mcmc:
        args2 = optimum_point, gen_points, val_points, dfs, scaler, X_train_scaled, y_train
        post_de_mcmc_args = application, daytype, num_layers, NN_model, Data, test_data, x_2023, error_metric,\
                                models,ycol, DE_model, NP_indices, reg_flag, return_method, print_master, NP, MCMC, DE_model
        models, data = post_DE_MCMC(post_de_mcmc_args, args2, mcmc_chain,DE_model.g)

    #master.append(dfs)                
    #run = run + 1
    #return dfs
    models=pd.concat(models,sort=False)
    #data=pd.concat(data,sort=False)
    return dfs, models


# population best

# path = r'/home/wesley/repos/DE/output/DE-NN-load-summary-standard-error-exploration-load-search-468-1500-refinement-bag-layers-weekday.ods'
# data = read_ods(path, sheet='model_data_mean')

# #data['B'] = '2023_RMSE_mean'
# data['B'] = pd.to_numeric(data['2023_RMSE_mean'])
# cols = ['NP_', 'G_', 'error_metric_']
# #test = data.loc[data.groupby(cols).B.idxmin()]
# test = data.loc[data.groupby(cols)['B'].idxmin()]

# out_dir = r'../output'
# output_name = 'agg'
# output_loc = os.path.join(out_dir + os.sep + output_name + '.csv')
# logging.info(f'Saving to {output_loc}')
# test.to_csv(output_loc)


if False:

    path = r'/home/wesley/repos/DE/output/paper-denn/DE-NN-load-summary-exploration-load-search-4-500-refinement-bag-weekend.ods'
    df = read_ods(path, sheet='model_data')
    key = ['layers', 'error_metric', 'Run']
    #df = df[df['2023_RMSE'] < 8].copy()
    data = df.groupby(key)["2023_RMSE"].min()
    data = data.reset_index(drop = False)
    #data.hist(column='2023_RMSE', by='error_metric', bins='auto')

    # output summary data

    kfc = (f'DE-NN-explore') 
    out_dir = r'../output'
    output_name = '-'.join(kfc)
    output_loc = os.path.join(out_dir + os.sep + output_name + '.ods')
    logging.info(f'Saving to {output_loc}')

    with pd.ExcelWriter(output_loc ) as writer:
        df.to_excel(writer, sheet_name = 'df', index=False)
        data.to_excel(writer, sheet_name = 'data', index=False)


# histogram of averages

# path = r'/home/wesley/repos/DE/output/paper-denn/DE-NN-load-summary-exploration-load-search-4-500-refinement-bag-weekend.ods'
# df = read_ods(path, sheet='model_data_mean')
# data = df.groupby("layers_")["2023_RMSE_mean"].min()
# data = data.reset_index(drop = False)
# data.hist(bins='auto')

# if False:

#     path = r'/home/wesley/repos/DE/output/paper-denn/DE-NN-load-summary-exploration-load-search-4-500-refinement-bag-weekend.ods'
#     df = read_ods(path, sheet='model_data')
#     key = ['layers', 'error_metric', 'Run']
#     #df = df[df['2023_RMSE'] < 8].copy()
#     data = df.groupby(key)["2023_RMSE"].min()
#     data = data.reset_index(drop = False)
#     #data.hist(column='2023_RMSE', by='error_metric', bins='auto')

#     # output summary data

#     kfc = (f'DE-NN-explore') 
#     out_dir = r'../output'
#     output_name = '-'.join(kfc)
#     output_loc = os.path.join(out_dir + os.sep + output_name + '.ods')
#     logging.info(f'Saving to {output_loc}')

#     with pd.ExcelWriter(output_loc ) as writer:
#         df.to_excel(writer, sheet_name = 'df', index=False)
#         data.to_excel(writer, sheet_name = 'data', index=False)
    
# if False:

#     path = r'/home/wesley/repos/DE/output/paper-denn/MLP-NN-kWh-PepperCanyon-weekend.ods'
#     df = read_ods(path, sheet='runs')
#     key = ['hidden_layer_sizes', 'batch_size', 'validation_fraction']
#     df = df[df['2019_RMSE'] < 8].copy()
#     data = df.groupby(key)["2019_RMSE"].min()
#     data = data.reset_index(drop = False)
#     #data.hist(column='2019_RMSE', bins='auto')
#     data.hist(column='2019_RMSE')


# path = r'/home/wesley/repos/DE/output/paper-denn/DE-NN-lmp-summary-standard-error-exploration-lmp-layers-rmsle.ods'
# df = read_ods(path, sheet='model_data')
# df['Run'] = df['Run'].astype(int)
# df = df[df['c'] == 1].copy()

# key_col = 'layers'
# keys = df[key_col].drop_duplicates().tolist()
                     
# c1 = 'TrainMin'
# c2 = '2023_RMSE'

# df_list = []

# for k in keys:
#         current = df[df[key_col] == k].copy()
#         cv = current[c1].corr(current[c2])
#         data = pd.DataFrame({'Run':[k], 'Correlation':[cv] })
#         df_list.append(data)
        
# test = pd.concat(df_list, sort=False)
# boo = False

# ffs = ['best', 'random', 'best2', 'random2', 'best3', 'random3']
# r = 8
# three = return_operator_weight(ffs, 3)

# ffs = ['best', 'random', 'best2', 'random2',]
# r = 3
# two = return_operator_weight(ffs, r)

# ffs = ['best', 'random', ]
# r = 3
# one = return_operator_weight(ffs, r)

#ffs = [ True, False]
#r = 4
#one = return_combo_list(ffs, r)

#ffs = [ 1,2,3,4,5]
#r = 3
#one = return_combo_list(ffs, r)
#print(f'length {len(one)}')

# ffs = ['mae', 'rmsle', 'log_cosh', 'r2', 'rae', 'mape', 'med_abs', 'rmse']

# test = return_error_index(ffs, 4, 1)
# test2 = return_error_index(ffs, 5, 1)
# test3 = return_error_index(ffs, 6, 1)
# test4 = return_error_index(ffs, 7, 1)
# test5 = return_error_index(ffs, 3, 1)
# test6 = return_error_index(ffs, 2, 1)

#ffs = [5,10]
#test = return_combo_list(ffs, 3)

# if vary_error:

#     # uniformly choose an error function.

#     loss_list = ['rmse', 'mae', 'mape', 'rmsle', 'rae', 'log_cosh', 'med_abs', 'r2']
#     error_function = random.choice(loss_list)

#     # create standard error dict with this choice

#     error_metric_dict = create_error_metric_dict(None, error_function, NP, 'uniform')

#     # set min value index to default

#     error_metric_dict[mindex] = DE_model.error_metric

if False:

    path = r'/home/wesley/repos/DE/output/DE-NN-lmp-refinement.csv'
    df = pd.read_csv(path)

    key_col = 'Run'
    keys = df[key_col].drop_duplicates().tolist()
                        
    c1 = 'Acceptance'
    c2 = '2023_RMSE'

    df_list = []

    plt.figure(figsize=(12,6))
    for k in keys:
            current = df[df[key_col] == k].copy()
            current = current.reset_index(drop = True)
            np_ = current.loc[0,'NP']
            x_ = np.arange(0,len(current))
            
            plt.plot(x_, current[c1],  label=f'Run {k}', linewidth=0.75)        
            plt.xlabel('Gen')
            plt.ylabel(c1)
            plt.legend(fontsize = 10, loc='upper right')
    plt.show()

    dir_path = '/home/wesley/repos/'
    kfc = ('AR', '15')
    output_name = '-'.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    #plt.close()

# import random
# import numpy as np

# def create_arrays():
#     arrays = []
#     for i in range(10):
#         arrays.append(np.random.rand(1, 2000))
#     return arrays

# import matplotlib.pyplot as plt

# def plot_arrays(arrays):
#     plt.figure(figsize=(10, 5))
#     plt.plot(np.mean(arrays, axis=0)[0], label='Mean')
#     plt.fill_between(range(2000), np.percentile(arrays, 5, axis=0)[0], np.percentile(arrays, 95, axis=0)[0], alpha=0.5, label='5% and 95% confidence interval')
#     plt.legend()
#     plt.show()

# arrays = create_arrays()
# plot_arrays(arrays)

if False:

    path = r'/home/wesley/repos/DE/output/paper-bayes-denn/weekend/DE-NN-load-summary-standard-error-exploration-load-search-8-15k-denn--weekend-parallel.ods'
    path2 = r'/home/wesley/repos/DE/output/paper-bayes-denn/weekday/DE-NN-load-summary-standard-error-exploration-load-search-8-15k-denn-parallel-weekday.ods'
    df = read_ods(path, sheet='model_data')
    df2 = read_ods(path2, sheet='model_data')

    fig, (ax1, ax2) = plt.subplots(2,figsize=(8, 6))
    #plt.figure(figsize=(18, 6))
    #fig.suptitle('Vertically stacked subplots')
    ax1.plot(df['c'], df['2023_rmse_MCMC_mode'])
    ax1.plot(df['c'], df['2023_RMSE_MCMC_mean'])
    ax1.set_title('Weekend')
    ax1.set(ylabel='rmse')
    ax1.legend(('MCMC_mode', 'MCMC_mean'), loc='lower left', shadow=True)
    ax2.plot(df2['c'], df2['2023_rmse_MCMC_mode'])
    ax2.plot(df2['c'], df2['2023_RMSE_MCMC_mean'])
    ax2.set_title('Weekday')
    ax2.set(xlabel='Chain', ylabel='rmse')
    ax2.legend(('MCMC_mode', 'MCMC_mean'), loc='upper right', shadow=True)
    plt.tight_layout()
    plt.show()

if False:

    path = r'/home/wesley/repos/DE/output/paper-denn/weekday/DE-NN-load-summary-standard-error-exploration-load-search-8-2k-denn-comparison-weekday.ods'
    df = read_ods(path, sheet='model_data')
    df = df[df['return_method'] == 'bma'].copy()
    df = df[df['bootstrapping'] == '(True, 1)'].copy()
        #df = df[df['init'] == ''].copy()
    df = df[df["init"].str.contains("uniform")]

    key_col = 'Run'
    keys = df[key_col].drop_duplicates().tolist()
                        
    c1 = '2023_RMSE'
    #c2 = '2023_RMSE'

    df_list = []

    plt.figure(figsize=(12,6))
    for k in keys:
            current = df[df[key_col] == k].copy()
            current = current.reset_index(drop = True)
            np_ = current.loc[0,'NP']
            x_ = np.arange(0,len(current))
            
            plt.plot(x_, current[c1],  label=f'Run {k}', linewidth=0.75)        
            plt.xlabel('Gen')
            plt.ylabel(c1)
            plt.legend(fontsize = 10, loc='upper right')
    plt.show()

    dir_path = '/home/wesley/repos/'
    kfc = ('AR', '15')
    output_name = '-'.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    #plt.close()


if False:

    path = r'/home/wesley/repos/DE/output/paper-denn/weekend/prelim/DE-NN-load-summary-standard-error-exploration-load-search-28-while-denn-rmsle-bma-val-sample-CURVE-even-weekend.ods'
    df = read_ods(path, sheet='model_data_mean')
    x = df['val_sample_']
    y = df['2023_RMSE_mean']
    #import matplotlib as mpl
    
    plt.figure(figsize=(12,6))
    plt.plot(x, y,  label='NP=28', linewidth=1)
    plt.title('Weekend BMA 20-Run Average', fontsize = 20,)
    plt.xlabel('Number of Validation Indices', fontsize = 20,)
    plt.ylabel('RMSE', fontsize = 20,)
    plt.legend(fontsize = 20, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    dir_path = '/home/wesley/repos/'
    kfc = ('Weekend', 'BMA Val Sample Curve')
    output_name = ' '.join(kfc)
    output_loc = os.path.join(dir_path + os.sep, output_name + '.png')
    logging.info(f'Saving to {output_loc}')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    plt.show()


# plot difference vectors

if False:

# Define the vectors

    a = np.array([2,6])
    b = np.array([4,4])
    base = np.array([10,3])
    diff = a-b
    test = base + diff

    d = np.array([6,5])
    e = np.array([8,9])
    base2 = np.array([1,3])
    diff2 = e-d

    plt.figure(figsize=(10,6))
    # Plot the first vector (v1) as an arrow
    plt.quiver(b[0], b[1], diff[0], diff[1], angles="xy", scale_units="xy", scale = 1, color="Red", width=2e-3)
    # Plot the second vector (v2) as an arrow
    plt.quiver(base[0], base[1], diff[0], diff[1], angles="xy", scale_units="xy", scale = 1, color="cornflowerblue", width=2e-3)

    # plot points

    plt.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale = 1, color="darkgreen", width=2e-3)
    plt.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale = 1, color="gray", width=2e-3)
    plt.quiver(0, 0, base[0], base[1], angles="xy", scale_units="xy", scale = 1, color="darkmagenta", width=2e-3)
    plt.quiver(0, 0, test[0], test[1], angles="xy", scale_units="xy", scale = 1, color="maroon", width=2e-3)

    plt.plot(a[0], a[1], '.', markersize=10, label='a')
    plt.plot(b[0], b[1], '.', markersize=10, label='b')
    plt.plot(base[0], base[1], '.', markersize=10, label='r1')
    plt.plot(test[0], test[1], '.', markersize=10, label='test')
    #plt.plot(d[0], d[1], '.', markersize=10, label='d')
    #plt.plot(e[0], e[1], '.', markersize=10, label='e')

    # Set the limits for the plot
    plt.xlim([0,11])
    plt.ylim([0,7])

    # annotate
    plt.annotate('r2', # this is the text
                    (a[0],a[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    fontsize=15,
                    ha='center') # horizontal alignment can be left, right or center

    plt.annotate('r3', # this is the text
                    (b[0],b[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    fontsize=15,
                    ha='center') # horizontal alignment can be left, right or center

    plt.annotate('r1', # this is the text
                    (base[0],base[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,15), # distance from text to points (x,y)
                    fontsize=15,
                    ha='center') # horizontal alignment can be left, right or center

    plt.annotate('r2-r3', # this is the text
                    (b[0],b[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(-35,50), # distance from text to points (x,y)
                    fontsize=15,
                    ha='center') # horizontal alignment can be left, right or center

    plt.annotate('r2-r3', # this is the text
                    (base[0],base[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(-35,50), # distance from text to points (x,y)
                    fontsize=15,
                    ha='center') # horizontal alignment can be left, right or center

    plt.annotate('r', # this is the text
                    (test[0],test[1]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(-10,10), # distance from text to points (x,y)
                    fontsize=15,
                    ha='center') # horizontal alignment can be left, right or center

    # Set the labels for the plot
    plt.xlabel('x', fontsize = 20,)
    plt.ylabel('y', fontsize = 20,)

    # Show the grid lines
    plt.grid()
    plt.show()

    dir_path = '/home/wesley/repos/'
    output_name = 'difference_vector'
    output_loc = os.path.join(dir_path + os.sep + 'images', output_name + '.png')
    plt.savefig(output_loc, dpi=300, bbox_inches = 'tight')
    #plt.close()

