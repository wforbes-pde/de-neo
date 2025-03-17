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
from DE_vector_helper import process, post_DE
from pandas_ods_reader import read_ods

print_master = False

def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    #logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s %(levelname)s %(message)s')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    start = datetime.now()
    np.random.seed(42)

    # DE parameter exploration

    e_ = 's'
    title = f'rosenbrock-exploration-{e_}'
    ffs = [3,4,5] # n^3
    #ffs = [10,20,30,40]

    DE_grid = {'G':[ 4000, ], # 2000
            'NP':[ 16 ] , # 4,4,4,4,4,4,4,4,4,4,  6,6,6,6,6,6,6,6,6,6,  8,8,8,8,8,8,8,8,8,8, 16,16,16,16,16,16,16,16,16,16, 12,12,12,12,12,12,12,12,12,12, 10,10,10,10,10,10,10,10,10,10,
            'F':[ 0.9, ], # 0.9, 0.7, 0.5 LARGER 0.9 SEEMS TO LEAD TO MEAN-VALUE FORECASTS!!!!!!!!!!!!!!!!!!!!
            'CR': [ 0.7 ], # 0.7
            'mutation_type': [ 'random', ], # random2 weekend
            'NPI': [ 4 ], # 4
            'track_len': [ 2 ], # min 2 
            'tol': [ 0, ], # -1e-3,
            'init': [ # ('normal',10,1),
                      #('uniform',-5,10),
                      # ('halton',None,None), # weekday
                       ('latin',None,10), # weekend
                      #  ('skunk',None,None),
                    ], 
            'refine_param': [# (10,10,10,False), 
                              (10,10,10,True), 
                             # (10,10,10,True), 
                                ], # refine_gen_start, refine_current_start, refine_mod_start, refine_random
            'F_refine': [ 'default',], # 'default', 'variable', 'weight_variable',
            'F_delta': [ 0.01 ], #
            'lowerF': [ 0.1, ], # 0.1
            'upperF': [ 0.9, ], # 0.9
            'mutation_refine': [ 'default', ], # 'default', 'variable', 'dimension_variable', 'candidate_variable', 'full_variable',
            'CR_refine': [ 'default', ], # 'default', 'variable', 'weight_variable', 
            'CR_delta': [0.1], # 
            'lowerCR': [ 0.1 ], # 0.1
            'upperCR': [ 0.9 ], # 0.9
            'return_method': [ 'standard', ], # 'bma', 'bma_val', 'standard', 'standard_val', 'bagging', 'bumping'
            'error_metric': [ 'rmsle' ], # 'rmsle', 'rmse', 'r2',
            'run_enh':  [ # run_svd, run_cluster, run_local
                             # (True, True, True),
                               (True, True, True),
                             #  (False, True, True),
                            ],
            'exhaustive': [  (False,20,6), 
                            # (True,50,6), 
                                 ], # run_exh, current, subset
            'val_sample': [ 4, ], # bma val minimum indices
            'd': [ 20, ], # dimension
            'test_function': [ 'rosenbrock' ], # test function to find minimum of
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
        init = param[8]
        refine_param = param[9]
        F_refine = param[10]
        F_delta = param[11]
        lowerF = param[12]
        upperF = param[13]
        mutation_refine = param[14]
        CR_refine = param[15]
        CR_delta = param[16]
        lowerCR = param[17]
        upperCR = param[18]
        return_method = param[19]                              
        error_metric = param[20]
        run_enh = param[21]       
        exhaustive_ = param[22]
        val_sample_ = param[23]
        d_ = param[24]
        test_function_ = param[25]

        DE_model = DEModelClass(NP, G, F, CR, mutation_type, tol, NPI, init, track_length,
                                F_refine, F_delta, lowerF, upperF,
                                mutation_refine, refine_param, 
                                CR_refine, CR_delta, lowerCR, upperCR,
                                return_method, error_metric, run_enh, exhaustive_, val_sample_,
                                run, d_, test_function_)

        # run DE and return resultant best-fitting candidate

        optimum_point, gen_points, val_points, dfs = differential_evolution_vector(DE_model)

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
    mdata = pd.DataFrame(mdata)
    # runtime

    time_taken = datetime.now() - start
    print(f'runtime was {time_taken}')
    
    # output full data

    kfc = (f'rosenbrock', 'refinement')
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
           'return_method', 'track_len', 'error_metric', 'run_enh',  'init', 'exh', 'val_sample',
           'c'] # add neurons for 1 layer ADD BACK 'c'
    key_cols = ['Minimum']
    
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
    perf_m2 = mdata.groupby(key2)[key_cols2].aggregate(['mean','count', 'min'])
    perf_m2 = perf_m2.reset_index(drop=False)

    # merge columns

    perf_m.columns = perf_m.columns.map('_'.join)
    perf_m2.columns = perf_m2.columns.map('_'.join)

    # refine count

    ref_count = return_refine_count(full_data)

    # group minimum

    perf_m['Minimum_mean'] = pd.to_numeric(perf_m['Minimum_mean'])
    cols = ['NP_', 'G_', 'error_metric_',]
    test = perf_m.loc[perf_m.groupby(cols)['Minimum_mean'].idxmin()]

    # output summary data

    kfc = (f'rosenbrock', 'summary', f'{title}') 
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
    