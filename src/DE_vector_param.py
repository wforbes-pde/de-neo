import pandas as pd
import itertools
import os
import sys
from datetime import datetime
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from DE_vector_helper import DEModelClass
from DE_vector_helper import return_refine_count
from DE_vector import differential_evolution_vector
from sklearn.neural_network import MLPRegressor
from DE_vector_helper import process, post_DE, post_DE_MCMC
from pandas_ods_reader import read_ods
from collections import Counter
import random

print_master = False

def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    #logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s %(levelname)s %(message)s')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    start = datetime.now()
    np.random.seed(42)

    # DE parameter exploration

    e_ = 'search-8-mcmc-large-bma-val-exp4'
    title = f'standard-error-exploration-{e_}'
    ffs = [3,4,5] # n^3
    #ffs = [10,20,30,40]

    DE_grid = {'G':[ 2000, ], # 2000
            'NP':[ 8,8,8,8,8 ] , # 4,4,4,4,4,4,4,4,4,4,  6,6,6,6,6,6,6,6,6,6,  8,8,8,8,8,8,8,8,8,8, 16,16,16,16,16,16,16,16,16,16, 12,12,12,12,12,12,12,12,12,12, 10,10,10,10,10,10,10,10,10,10,
            'F':[ 0.9, ], # 0.9, 0.7, 0.5 LARGER 0.9 SEEMS TO LEAD TO MEAN-VALUE FORECASTS!!!!!!!!!!!!!!!!!!!!
            'CR': [ 0.7 ], # 0.7
            'mutation_type': [ 'random', ], # random2 weekend
            'NPI': [ 4 ], # 4
            'track_len': [ 2 ], # min 2 
            'tol': [ 0 ], # -1e-3,
            'train_size': [  (1,False,1,None),
                                # (0.5, True, 1000, None), 
                           ], # 0.7 train_data_size, run_val, val_gen_min, val_metric
            'init': [ # ('he',None,None),
                      ('uniform',-5,10),
                     #  ('halton',None,None), # weekday
                      # ('latin',None,None), # weekend
                      #  ('skunk',None,None),
                    ], 
            'refine_param': [ # (10,10,10,False), 
                             # (10,10,10,True), 
                              (10,10,10,True), 
                                ], # (100,2,10) refine_gen_start, refine_current_start, refine_mod_start, refine_random
                            # return_combo_list(ffs, 2),
            'F_refine': [ 'default',], # 'default', 'variable', 'weight_variable',
            'F_delta': [ 0.01 ], #
            'lowerF': [ 0.1, ], # 0.1
            'upperF': [ 0.9, ], # 0.9
            'mutation_refine': [ 'default', ], # 'default', 'variable', 'weight_variable',
            'CR_refine': [ 'default', ], # 'default', 'variable', 'weight_variable', 
            'CR_delta': [0.1], # 
            'lowerCR': [ 0.1 ], # 0.1
            'upperCR': [ 0.9 ], # 0.9
            'return_method': [ 'bma_val', ], # 'bma', 'bma_val', 'standard', 'standard_val', 'bagging', 'bumping'
            'error_metric': [ 'rmsle' ], # 'rmsle', 'rmse', 'r2',
            'run_enh':  [ # run_svd, run_cluster, run_local
                              (True, True, True),
                             #  (False, False, False),
                             #  (False, True, True),
                            ],
                          # return_combo_list([True, False], 3),
            'bootstrapping': [  (True, 1,), 
                               # (False, 1),
                        ], # bootstrap samples for each index training. needed for bagging and bumping. optional for standard and bma.
            'exhaustive': [  (False,20,6), 
                            # (True,50,6), 
                                 ], # run_exh, current, subset
            'val_sample': [ 4, ], # bma val minimum indices
            'd': [ 5, ], # dimension
            'test_function': [ 'rosenbrock' ], # dimension
    }
    
    a = DE_grid.values()
    combinations = list(itertools.product(*a))
    
    # forecasting

    models = []
    master = []
    run = 0
    result_ids = []
    for param in combinations:
        #try:
        logging.info(f'Starting DE exploration Run {run}/{len(combinations)} {param}')

        G = param[0] # max number of generations
        NP = param[1] # number of parameter vectors in each generation         
        F = param[2] # mutate scaling factor
        CR = param[3] # crossover rate 
        mutation_type = param[4]
        NPI = param[5]
        track_length = param[6]
        tol = param[7]
        train_size = param[8]
        init = param[9]
        refine_param = param[10]
        F_refine = param[11]
        F_delta = param[12]
        lowerF = param[13]
        upperF = param[14]
        mutation_refine = param[15]
        CR_refine = param[16]
        CR_delta = param[17]
        lowerCR = param[18]
        upperCR = param[19]
        return_method = param[20]                              
        error_metric = param[21]
        run_enh = param[22]       
        bootstrapping = param[23]
        exhaustive_ = param[24]
        val_sample_ = param[25]
        d_ = param[26]
        test_function_ = param[27]

        DE_model = DEModelClass(NP, G, F, CR, mutation_type, tol, NPI, init, track_length,
                                F_refine, F_delta, lowerF, upperF,
                                mutation_refine, refine_param, 
                                CR_refine, CR_delta, lowerCR, upperCR,
                                return_method, error_metric, run_enh, bootstrapping, exhaustive_, val_sample_,
                                run, d_, test_function_)

        # run DE and return resultant best-fitting candidate

        optimum_point, gen_points, val_points, dfs = differential_evolution_vector(DE_model, train_size)

        # model selection

        NP_indices = list(np.arange(0,NP))

        args2 = optimum_point, gen_points, val_points, dfs
        post_de_args = error_metric, models,DE_model, NP_indices, return_method, print_master, NP, DE_model
        models, data = post_DE(post_de_args, args2)
        
        master.append(dfs)
        run = run + 1
    #except:

    # collect results

    full_data = pd.concat(master, sort=False)
    mdata = pd.concat(models, sort=False)

    # runtime

    time_taken = datetime.now() - start
    print(f'runtime was {time_taken}')
    
    # output full data

    kfc = (f'DE-NN-{application}', 'refinement')
    out_dir = r'../output'
    output_name = '-'.join(kfc)
    output_loc = os.path.join(out_dir + os.sep + output_name + '.csv')
    #logging.critical(f'Saving to {output_loc}')
    #full_data.to_csv(output_loc)
    
    # slice exit generation

    for_agg = full_data[full_data['Exit'] == 'True' ].copy()

    # summary standard average

    key = ['G', 'NP', 'NPI', 'F', 'CR', 'mutation_type', 'tol', 
           'F_delta', 'lowerF', 'upperF', 'F_refine', 'refine_param', 
           'mutation_refine', 'lowerCR', 'upperCR', 'CR_refine', 'CR_delta', 
           'run_mcmc', 'burn_in', 'return_method', 'track_len', 'pred_post_sample',
           'error_metric', 'reg_flag', 'run_enh', 'regularization_', 
           'error_dist', 'error_std', 'init', 'exh', 'val_sample',
           'bootstrapping', 'layers', 'train_size', 'Activation', 'c'] # add neurons for 1 layer ADD BACK 'c'
    if MCMC.run_mcmc:
        key_cols = [f'2023_{error_metric}_MCMC_mode', '2023_RMSE_MCMC_mode', 'TestStd', '2023_RMSE_MCMC_mean']
        #key.remove('c')
    else:
        key_cols = [f'2023_{error_metric}', '2023_RMSE', 'TestStd']
    
    # across each group and index c
    mdata['c'] = mdata['c'].astype(str)

    perf_m = mdata.groupby(key)[key_cols].aggregate(['mean','count', 'min'])
    perf_m = perf_m.reset_index(drop=False)

    # what about minimum across a run irrespective of index?
    # i.e. minimum at c=1 one run, then minimum at c=2 at the end.

    key2 = key.copy()
    #key2.remove('c')
    key2.append('Run')
    key_cols2 = key_cols.copy()
    key_cols2.remove('TestStd') # add this?
    perf_m2 = mdata.groupby(key2)[key_cols2].aggregate(['mean','count', 'min'])
    perf_m2 = perf_m2.reset_index(drop=False)

    discard = ['F_W0', 'F_W1', 'F_W2', 'F_W3', 'F_b0', 'F_b1', 'F_b2', 'F_b3',
                        'F2_W0', 'F2_W1', 'F2_W2', 'F2_W3', 'F2_b0', 'F2_b1', 'F2_b2','F2_b3']
    mdata = mdata[mdata.columns[~mdata.columns.isin(discard )]]

    # merge columns

    perf_m.columns = perf_m.columns.map('_'.join)
    perf_m2.columns = perf_m2.columns.map('_'.join)

    # refine count

    ref_count = return_refine_count(full_data)

    # group minimum

    if MCMC.run_mcmc:
        perf_m['2023_RMSE_MCMC_mode_mean'] = pd.to_numeric(perf_m['2023_RMSE_MCMC_mode_mean'])
        cols = ['NP_', 'G_', 'error_metric_', ]
        test = perf_m.loc[perf_m.groupby(cols)['2023_RMSE_MCMC_mode_mean'].idxmin()]

    if not MCMC.run_mcmc:
        perf_m['2023_RMSE_mean'] = pd.to_numeric(perf_m['2023_RMSE_mean'])
        cols = ['NP_', 'G_', 'error_metric_',]
        test = perf_m.loc[perf_m.groupby(cols)['2023_RMSE_mean'].idxmin()]

    # output summary data

    kfc = (f'DE-NN-{application}', 'summary', f'{title}', f'{daytype}') 
    out_dir = r'../output'
    output_name = '-'.join(kfc)
    output_loc = os.path.join(out_dir + os.sep + output_name + '.ods')
    logging.critical(f'Saving to {output_loc}')

    with pd.ExcelWriter(output_loc ) as writer:
        for_agg.to_excel(writer, sheet_name = 'training_exit', index=False)
        mdata.to_excel(writer, sheet_name = 'model_data', index=False)
        perf_m.to_excel(writer, sheet_name = 'model_data_mean')
        test.to_excel(writer, sheet_name = 'model_data_mean_min')
        ref_count.to_excel(writer, sheet_name = 'refine_count')
        perf_m2.to_excel(writer, sheet_name = 'test')

if __name__ == '__main__':
    main(sys.argv[1:])
    