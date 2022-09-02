import numpy as np
from sklearn.model_selection import ParameterGrid
from time import time
from .MethylationHMM import MethylationHMM
import pickle
import json
import os


class GridSearchHMM(object):
    
    def __init__(
        self,
        n_states = [3],
        init_types = ['random', 'uniform'],
        distance_params = [75],
        learning_rates = [0.5],
        n_init = 1,
        n_iter = [100],
        conv_thresh = [0.1],  
        conv_iter = [2],
        n_processes = 1,
        workdir = '.'
    ):

        self.n_processes = n_processes
        self.workdir = workdir
        self._n_init = n_init
        self._grid = list(ParameterGrid(
            {'n_states': n_states, 
             'init_type': init_types, 
             'distance_param': distance_params,  
             'learning_rate': learning_rates, 
             'n_iter': n_iter, 
             'conv_thresh': conv_thresh, 
             'conv_iter': conv_iter, 
            }
        ))
        self._results = []
        self.best_hmm = None
        self._id = None
        
    def fit(self, train_set, test_set, to_file=False):
        if type(train_set) is str:
            with open(train_set, 'rt') as fp:
                tmp_set = json.load(fp)
                train_set = []
                for t in tmp_set:
                    train_set += [np.array(t)]
                
        if type(test_set) is str:
            with open(test_set, 'rt') as fp:
                tmp_set = json.load(fp)
                test_set = []
                for t in tmp_set:
                    test_set += [np.array(t)]
        
        start = time()
        for i, params_set in enumerate(self._grid):
            tmp_hmm = self._init_hmm(params_set)
            tmp_trained_hmm, tmp_history = self._train_one(tmp_hmm,params_set,train_set)
            tmp_prediction, _ = tmp_trained_hmm.decode(test_set, algorithm='viterbi')
            end = time() 
            self._results += [{
                'hmm': tmp_trained_hmm,
                'params': params_set,
                'train_history': tmp_history,
                'test_score': tmp_prediction,
                'time': end-start,
            }]
            if self.best_hmm is not None:
                self.best_hmm = self._results[-1] if \
                    self._results[-1]['test_score'] > self.best_hmm['test_score'] \
                    else self.best_hmm
            else:
                self.best_hmm = self._results[-1]
            print("Fitted {} models in {:.1f} seconds. Best log score: {}".format(i+1,
                                                                                  self._results[-1]['time'],
                                                                                  self.best_hmm['test_score']))
            if to_file:
                gs_id = self._id if self._id is not None else ''
                self.save_results(os.path.join(self.workdir,f'GridSearchHMM_{gs_id}.results'))
    
    def save_grid(self, file_name, n_clusters=1):
        if n_clusters > 1:
            for i in range(n_clusters):
                batch_size = len(self._grid)//n_clusters + (len(self._grid)%n_clusters > 0)
                tmp_data = self._grid[i*batch_size : (i+1)*batch_size]
                with open(file_name + f'_{i+1}.json', 'wt') as fp:
                    json.dump({'id': i, 'grid': tmp_data}, fp, indent='\t')
        else:
            with open(file_name, 'wt') as fp:
                json.dump({'grid': self._grid}, fp, indent='\t')
            
    def load_grid(self, file_name):
        with open(file_name, 'rt') as fp:
            tmp_data = json.load(fp)
            tmp_grid = tmp_data.get('grid', None)
            self._grid = tmp_grid if tmp_grid is not None else self._grid
            self._id = tmp_data.get('id', None)
            
    def save_results(self, file_name):
        if not os.path.isdir(file_name + '.models'):
            os.mkdir(file_name + '.models', 0o777)
        with open(file_name, 'wt') as fp:
            tmp_results = []
            for i,r in enumerate(self._results):
                tmp_dict = {}
                for k in r:
                    if k!='hmm':
                        tmp_dict[k] = r[k]
                    else:
                        tmp_dict[k] = os.path.join(file_name+".models",f'{self._id}_{i}.hmm')
                        model = r[k]
                        with open(tmp_dict[k], 'wb') as f:
                            pickle.dump(model, f)
                tmp_results += [tmp_dict]
            #print(tmp_results)
            json.dump(tmp_results, fp, indent='\t')
            
    
    def _train_one(self, hmm, training_params, train_seq):
        trained_hmm, log_likelihoods = hmm.train(
            train_seq,
            n_init=self._n_init,
            n_iter=training_params['n_iter'],
            conv_thresh=training_params['conv_thresh'],
            conv_iter=training_params['conv_iter'],
            ignore_conv_crit=False,
            plot_log_likelihood=False,
            no_init=False,
            n_processes=self.n_processes,
            print_every=1,
            return_log_likelihoods=True,
        )
        return trained_hmm, log_likelihoods
    
    def _init_hmm(self, init_params):
        hmm = MethylationHMM(
            init_params['n_states'],
            distance_param=init_params['distance_param'],
            tr_params='e',
            init_params='e',
            init_type=init_params['init_type'],
            pi_prior=1.0,
            A_prior=1.0,
            nr_no_train_de=0,
            state_no_train_de=None,
            learning_rate=init_params['learning_rate'],
            verbose=False,
        )
        return hmm