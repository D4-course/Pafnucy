'''
Split dataset into train and test set
'''
import os

import argparse
from sklearn.utils import shuffle
import h5py



parser = argparse.ArgumentParser(description='Split dataset int training,'
                                 ' validation and test sets.')
parser.add_argument('--input_path', '-i', default='../pdbbind/v2016/',
                    help='directory with pdbbind dataset')
parser.add_argument('--output_path', '-o', default='../pdbbind/v2016/',
                    help='directory to store output files')
parser.add_argument('--size_val', '-s', type=int, default=1000,
                    help='number of samples in the validation set')
args = parser.parse_args()

# create files with the training and validation sets
with h5py.File(f'{args.output}/training_set.hdf' , 'w') as g, \
     h5py.File(f'{args.output_path}/validation_set.hdf', 'w') as h:
    with h5py.File(f'{args.input_path}/refined.hdf', 'r') as f:
        refined_shuffled = shuffle(list(f.keys()), random_state=123)
        for pdb_id in refined_shuffled[:args.size_val]:
            ds = h.create_dataset(pdb_id, data=f[pdb_id])
            ds.attrs['affinity'] = f[pdb_id].attrs['affinity']
        for pdb_id in refined_shuffled[args.size_val:]:
            ds = g.create_dataset(pdb_id, data=f[pdb_id])
            ds.attrs['affinity'] = f[pdb_id].attrs['affinity']
    with h5py.File(f'{args.input_path}/general.hdf', 'r') as f:
        for pdb_id in f:
            ds = g.create_dataset(pdb_id, data=f[pdb_id])
            ds.attrs['affinity'] = f[pdb_id].attrs['affinity']

# create a symlink for the test set
os.symlink(os.path.abspath(f'{args.input_path}/core.hdf' ),
           os.path.abspath(f'{args.output_path}/test_set.hdf' ))
