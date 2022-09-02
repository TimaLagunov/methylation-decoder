from MethylationHMM.my_GridSearch import GridSearchHMM
import argparse
import os

def main(command_line=None):
    #####################################################
    #Parsing arguments block
    ###
    
    parser = argparse.ArgumentParser(description='Run GridSearch in parallel')
    parser.add_argument("work_dir", metavar="<work dir>", help="working directory for all outputs", type=str)
    parser.add_argument("seq_dir", metavar="<seq dir>", help="directory with train_set.seq and test_set.seq, made by set_maker.py", type=str)
    parser.add_argument("grid_params", metavar="<grid params path>", help="path to json file with grid parameters", type=str)
    parser.add_argument("-p","--processes", metavar="<N proc>", help="Number of processes (default: 1)", type=int, default=1)
    
    args = parser.parse_args(command_line)
    
    #####################################################
    #Script block
    ###
    
    work_dir = args.work_dir
    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)

    seq_dir = args.seq_dir
    grid_params = args.grid_params
    proc = args.processes
    
    grid_searcher = GridSearchHMM(n_processes = proc)
    grid_searcher.load_grid(grid_params)
    print(f'Start fitting!!!')
    grid_searcher.fit(os.path.join(seq_dir, 'train_set.seq'), os.path.join(seq_dir, 'test_set.seq'), to_file=True)
    
if __name__ == '__main__':
    main()