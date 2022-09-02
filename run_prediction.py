from MethylationHMM.MethylationHMM import MethylationHMM
import numpy as np
import pandas as pd
import pickle
import json
from set_maker import set_maker
import os

import argparse

def main(command_line=None):
    
    parser = argparse.ArgumentParser(description='Run HMM for state decoding')
    parser.add_argument("work_dir", metavar="<work dir>", help="working directory for all outputs", type=str)
    parser.add_argument("train_file", metavar="<train file>", help="path to train file (could be train.seq file or CpG_report file)", type=str)
    parser.add_argument("pred_file", metavar="<dec file>", help="path to CpG_report file that you want to decode", type=str)
    parser.add_argument("-n","--n_states", metavar="<int>", help="number of HMM states (default: 3)", type=int, default=3)
    parser.add_argument("-dp","--distance_param", metavar="<int>", help="distance parameter for HMM (default: 1250)", type=int, default=1250)
    parser.add_argument("-lr","--learning_rate", metavar="<float>", help="learning_rate for HMM (default: 0.1)", type=float, default=0.1)
    parser.add_argument("-ni","--n_iter", metavar="<int>", help="maximum number of iterations in training (default: 1000)", type=int, default=1000)
    parser.add_argument("-ct","--conv_thresh", metavar="<float>", help="the threshold for the likelihood increase (convergence) (default: 1e-4)", type=float, default=1e-4)
    parser.add_argument("-ci","--conv_iter", metavar="<int>", help="number of iterations for which the convergence criteria has to hold before early-stopping; (default: 100)", type=int, default=100)
    parser.add_argument("-p","--processes", metavar="<int>", help="number of processes (default: 1)", type=int, default=1)
    parser.add_argument("-v","--verbosity", metavar="<int>", help="print every <int> interation log (default: 1)", type=int, default=1)
    
    args = parser.parse_args(command_line)
    
    workdir = args.work_dir
    train_file = args.train_file
    pred_file = args.pred_file
    n_states = args.n_states
    distance_param = args.distance_param
    learning_rate = args.learning_rate
    n_iter = args.n_iter
    conv_thresh = args.conv_thresh
    conv_iter = args.conv_iter
    proc = args.processes
    verb = args.verbosity
    
    model = MethylationHMM(
                n_states, 
                distance_param = distance_param,
                learning_rate = learning_rate,
            )
    
    if train_file.split('.')[-1] == 'seq':
        with open(train_file, 'rt') as fp:
                    tmp_set = json.load(fp)
                    train_set = []
                    for t in tmp_set:
                        train_set += [np.array(t)]
    else:
        _, train_set = set_maker(train_file)
    
    model_trained, _ = model.train(
                    train_set,
                    n_iter = n_iter,
                    conv_thresh = conv_thresh,
                    conv_iter = conv_iter,
                    n_processes = proc,
                    print_every = verb,
                    return_log_likelihoods=True,
                )
    
    positions = []
    depth = []
    positions_tmp = []
    methylated = []
    with open(pred_file, "rt") as file, \
    open(os.path.join(workdir,pred_file+'.prediction.bed'), 'wt') as pred_file, \
    open(os.path.join(workdir,pred_file+'.prediction_reverse.bed'), 'wt') as pred_rev_file, \
    open(os.path.join(workdir,pred_file+'.prediction_log'), 'wt') as log_file:
        line = file.readline()[:-1].split("\t")
        prev_chr = line[0]
        log_file.write(f'Chromosome\tscore\tscore reverse\tn_samples\n')
        while len(line) > 1:
            if line[0]==prev_chr:
                methylated += [int(line[3])]
                depth += [int(line[3])+int(line[4])]
                positions_tmp += [int(line[1])]
                line = file.readline()[:-1].split("\t")
            else:
                methylated = np.array(methylated).reshape((1,-1))
                depth = np.array(depth).reshape((1,-1))
                positions_tmp  = np.array(positions_tmp)
                positions = [positions_tmp,positions_tmp[::-1]]
                dist_plus = np.concatenate([[int(distance_param*10e3)],positions_tmp[1:] - positions_tmp[:-1]]).reshape((1,-1))
                dist_minus = np.concatenate([positions_tmp[1:] - positions_tmp[:-1],[int(distance_param*10e3)]]).reshape((1,-1))
                tmp_seq = []
                tmp_seq += [np.concatenate([methylated,depth,dist_plus,dist_minus]).T]
                rev_tmp_seq = np.concatenate([tmp_seq[0][::-1,0].reshape((1,-1)),
                                               tmp_seq[0][::-1,1].reshape((1,-1)),
                                               tmp_seq[0][::-1,3].reshape((1,-1)),
                                               tmp_seq[0][::-1,2].reshape((1,-1))]).T
                
                score, prediction = model_trained.decode(tmp_seq, algorithm='viterbi')
                score_rev, prediction_rev = model_trained.decode([rev_tmp_seq], algorithm='viterbi')
                
                log_file.write(f'{prev_chr}\t{score}\t{score_rev}\t{len(prediction[0])}\n')
                for i,p in enumerate(prediction[0]):
                    pred_file.write("".join(str(x)+" " for x in [prev_chr, 
                                                         positions[0][i], 
                                                         positions[0][i]+1, 
                                                         p,
                                                         tmp_seq[0][i][1],
                                                         '+',
                                                         positions[0][i], 
                                                         positions[0][i]+1,
                                                         f'{int(255/(n-1)*(n-1 - p))},0,{int(255/(n-1)*(p))}',
                                                        ])[:-1]+"\n")
                for i,p in enumerate(prediction_rev[0]):
                    pred_rev_file.write("".join(str(x)+" " for x in [prev_chr, 
                                                         positions[1][i], 
                                                         positions[1][i]+1, 
                                                         p,
                                                         [rev_tmp_seq][0][i][1],
                                                         '-',
                                                         positions[1][i], 
                                                         positions[1][i]+1,
                                                         f'{int(255/(n-1)*(n-1 - p))},0,{int(255/(n-1)*(p))}',
                                                        ])[:-1]+"\n")
                
                depth = []
                positions_tmp = []
                methylated = []
                
                prev_chr = line[0]
        
        methylated = np.array(methylated).reshape((1,-1))
        depth = np.array(depth).reshape((1,-1))
        positions_tmp  = np.array(positions_tmp)
        positions = [positions_tmp,positions_tmp[::-1]]
        dist_plus = np.concatenate([[int(distance_param*10e3)],positions_tmp[1:] - positions_tmp[:-1]]).reshape((1,-1))
        dist_minus = np.concatenate([positions_tmp[1:] - positions_tmp[:-1],[int(distance_param*10e3)]]).reshape((1,-1))
        tmp_seq = []
        tmp_seq += [np.concatenate([methylated,depth,dist_plus,dist_minus]).T]
        rev_tmp_seq = np.concatenate([tmp_seq[0][::-1,0].reshape((1,-1)),
                                       tmp_seq[0][::-1,1].reshape((1,-1)),
                                       tmp_seq[0][::-1,3].reshape((1,-1)),
                                       tmp_seq[0][::-1,2].reshape((1,-1))]).T
        
        score, prediction = model_trained.decode(tmp_seq, algorithm='viterbi')
        score_rev, prediction_rev = model_trained.decode([rev_tmp_seq], algorithm='viterbi'
        
        log_file.write(f'{prev_chr}\t{score}\t{score_rev}\t{len(prediction[0])}\n')
        for i,p in enumerate(prediction[0]):
            pred_file.write("".join(str(x)+" " for x in [prev_chr, 
                                                 positions[0][i], 
                                                 positions[0][i]+1, 
                                                 p,
                                                 tmp_seq[0][i][1],
                                                 '+',
                                                 positions[0][i], 
                                                 positions[0][i]+1,
                                                 f'{int(255/(n-1)*(n-1 - p))},0,{int(255/(n-1)*(p))}',
                                                ])[:-1]+"\n")
        for i,p in enumerate(prediction_rev[0]):
            pred_rev_file.write("".join(str(x)+" " for x in [prev_chr, 
                                                 positions[1][i], 
                                                 positions[1][i]+1, 
                                                 p,
                                                 [rev_tmp_seq][0][i][1],
                                                 '-',
                                                 positions[1][i], 
                                                 positions[1][i]+1,
                                                 f'{int(255/(n-1)*(n-1 - p))},0,{int(255/(n-1)*(p))}',
                                                ])[:-1]+"\n")
    
    with open(os.path.join(workdir,pred_file + '.hmm'), 'wb') as f:
        pickle.dump(model_trained, f)

if __name__ == '__main__':
    main()