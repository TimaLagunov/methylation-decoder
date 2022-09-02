import numpy as np
import json
import os

import argparse

def set_maker(file_path, subsamp_dict={}, to_file=False, file_name=None, out_dir=None):
    train_seq = []
    cpg_am_his = []
    with open(file_path, "rt") as file:
        line = file.readline()[:-1].split("\t")
        
        depth = []
        positions = []
        methylated = []
        i=0
        while len(line) > 1:
            if sum(x[0]<int(line[1])<x[1] for x in subsamp_dict.get(line[0],[(-1,-1)])) == 1 or len(subsamp_dict) == 0:
                methylated += [int(line[3])]
                depth += [int(line[3])+int(line[4])]
                positions += [int(line[1])]
                i+=1
            elif i>0:
                cpg_am_his += [i,i]
                
                methylated = np.array(methylated).reshape((1,-1))
                depth = np.array(depth).reshape((1,-1))
                positions  = np.array(positions)

                dist_plus = np.concatenate([[int(10e3)],positions[1:] - positions[:-1]]).reshape((1,-1))
                dist_minus = np.concatenate([positions[1:] - positions[:-1],[int(10e3)]]).reshape((1,-1))

                tmp_seq = []
                tmp_seq += [np.concatenate([methylated,depth,dist_plus,dist_minus]).T]
                rev_train_seq = np.concatenate([tmp_seq[0][::-1,0].reshape((1,-1)),
                                               tmp_seq[0][::-1,1].reshape((1,-1)),
                                               tmp_seq[0][::-1,3].reshape((1,-1)),
                                               tmp_seq[0][::-1,2].reshape((1,-1))]).T

                train_seq += tmp_seq + [rev_train_seq]
                
                depth = []
                positions = []
                methylated = []
                i=0
            
            line = file.readline()[:-1].split("\t")
            
    cpg_am_his += [i,i]
    
    methylated = np.array(methylated).reshape((1,-1))
    depth = np.array(depth).reshape((1,-1))
    positions  = np.array(positions)

    dist_plus = np.concatenate([[int(10e3)],positions[1:] - positions[:-1]]).reshape((1,-1))
    dist_minus = np.concatenate([positions[1:] - positions[:-1],[int(10e3)]]).reshape((1,-1))

    tmp_seq = []
    tmp_seq += [np.concatenate([methylated,depth,dist_plus,dist_minus]).T]
    rev_train_seq = np.concatenate([tmp_seq[0][::-1,0].reshape((1,-1)),
                                   tmp_seq[0][::-1,1].reshape((1,-1)),
                                   tmp_seq[0][::-1,3].reshape((1,-1)),
                                   tmp_seq[0][::-1,2].reshape((1,-1))]).T

    train_seq += tmp_seq + [rev_train_seq]
            
    if to_file:
        if out_dir is None:
            out_dir = os.path.split(file_path)[0]
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        if file_name is None:
            file_name = os.path.split(file_path)[1]+".seq"
        tmp_data = []
        for t in train_seq:
            tmp_data += [t.tolist()]
        with open(os.path.join(out_dir,file_name+".seq"), 'wt') as fp:
            json.dump(tmp_data, fp)
        
        return 0
    else:
        return cpg_am_his, train_seq
    
    
    
def main(command_line=None):
    parser = argparse.ArgumentParser(description='Prepare bismark CpG_report file to HMM and save')
    parser.add_argument("file_path", metavar="<file path>", nargs='+', help="path (or pathes) to CpG_report file from bismark", type=str)
    parser.add_argument("-fn", "--file_name", metavar="<f_name>", nargs='+',  help="name (or names) for output file(s) (defaul is the same as CpG_report file)", type=str)
    parser.add_argument("-o", "--out_dir", metavar="<out dir>",  help="output directory (defaul is directory with CpG_report file)", type=str)
    
    args = parser.parse_args(command_line)
    
    out_dir = args.out_dir if len(args.out_dir)>0 else None
    file_names = args.file_name if len(args.file_name)>0 else None
    file_pathes = args.file_path
    
    if len(file_pathes) > 1 and len(file_names) > 1:
        if len(file_pathes)!=len(file_names):
            print("If you use multiple file names AND multiple pathes, their amount should be the same")
            exit
        else:
            for i,rep_file in enumerate(file_pathes):
                set_maker(rep_file, to_file=True, file_name=file_names[i], out_dir=out_dir)
    else:
        for rep_file in file_pathes:
            for f_name in file_names:
                set_maker(rep_file, to_file=True, file_name=f_name, out_dir=out_dir)
    
    
if __name__ == '__main__':
    main()