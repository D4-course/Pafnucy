'''
Training set preparation for the model
'''

import os
import argparse
import numpy as np
import pandas as pd
import h5py
import pybel
from tfbio.data import Featurizer

def input_file(path):
    """Check if input file exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError(f'File {path} does not exist.')
    return path


def output_file(path):
    """Check if output file can be created."""

    path = os.path.abspath(path)
    dirname = os.path.dirname(path)

    if not os.access(dirname, os.W_OK):
        raise IOError(f'File {path} cannot be created (check your permissions).')
    return path


def string_bool(string):
    """Check if string is a boolean"""
    string = string.lower()
    if string in ['true', 't', '1', 'yes', 'y']:
        return True
    if string in ['false', 'f', '0', 'no', 'n']:
        return False
    raise IOError(f'{string} cannot be interpreted as a boolean')


parser = argparse.ArgumentParser(
    description='Prepare molecular data for the network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog='''This script reads the structures of ligands and pocket(s),
    prepares them for the neural network and saves in a HDF file.
    It also saves affinity values as attributes, if they are provided.
    You can either specify a separate pocket for each ligand or a single
    pocket that will be used for all ligands. We assume that your structures
    are fully prepared.\n\n

    Note that this scripts produces standard data representation for our network
    and saves all required data to predict affinity for each molecular complex.
    If some part of your data can be shared between multiple complexes
    (e.g. you use a single structure for the pocket), you can store the data
    more efficiently. To prepare the data manually use functions defined in
    tfbio.data module.
    '''
)

parser.add_argument('--ligand', '-l', required=True, type=input_file, nargs='+',
                    help='files with ligands\' structures')
parser.add_argument('--pocket', '-p', required=True, type=input_file, nargs='+',
                    help='files with pockets\' structures')
parser.add_argument('--ligand_format', type=str, default='mol2',
                    help='file format for the ligand,'
                         ' must be supported by openbabel')

parser.add_argument('--pocket_format', type=str, default='mol2',
                    help='file format for the pocket,'
                         ' must be supported by openbabel')
parser.add_argument('--output', '-o', default='./complexes.hdf',
                    type=output_file,
                    help='name for the file with the prepared structures')
parser.add_argument('--mode', '-m', default='w',
                    type=str, choices=['r+', 'w', 'w-', 'x', 'a'],
                    help='mode for the output file (see h5py documentation)')
parser.add_argument('--affinities', '-a', default=None, type=input_file,
                    help='CSV table with affinity values.'
                         ' It must contain two columns: `name` which must be'
                         ' equal to ligand\'s file name without extenstion,'
                         ' and `affinity` which must contain floats')
parser.add_argument('--verbose', '-v', default=True, type=string_bool,
                    help='whether to print messages')

args = parser.parse_args()


''' TODO: training set preparation (allow to read affinities) '''


num_pockets = len(args.pocket)
num_ligands = len(args.ligand)
if num_pockets not  in [1] and num_pockets != num_ligands:
    raise IOError(f'{num_pockets} pockets specified for {num_ligands} ligands. You must either'
                  ' provide a single pocket or a separate pocket for each ligand')
if args.verbose:
    print(f'{num_ligands} ligands and {num_pockets} pockets to prepare:')
    if num_pockets == 1:
        print(f' pocket: {args.pocket[0]}')
        for ligand_file in args.ligand:
            print(f' ligand: {ligand_file}' )
    else:
        for ligand_file, pocket_file in zip(args.ligand, args.pocket):
            print(f' ligand: {ligand_file}, pocket: {pocket_file}')
    print('\n\n')


if args.affinities is not None:
    affinities = pd.read_csv(args.affinities)
    if 'affinity' not in affinities.columns:
        raise ValueError('There is no `affinity` column in the table')
    if 'name' not in affinities.columns:
        raise ValueError('There is no `name` column in the table')
    affinities = affinities.set_index('name')['affinity']
else:
    affinities = None

featurizer = Featurizer()


def __get_pocket():
    if num_pockets > 1:
        for pocket_file in args.pocket:
            if args.verbose:
                print(f'reading {pocket_file}')
            try:
                pocket = next(pybel.readfile(args.pocket_format, pocket_file))
            except Exception as _:
                raise IOError(f'Cannot read {pocket_file} file')

            pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
            yield (pocket_coords, pocket_features)

    else:
        pocket_file = args.pocket[0]
        try:
            pocket = next(pybel.readfile(args.pocket_format, pocket_file))
        except Exception as _:
            raise IOError(f'Cannot read {pocket_file} file')
        pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
        for _ in range(num_ligands):
            yield (pocket_coords, pocket_features)


with h5py.File(args.output, args.mode) as f:
    pocket_generator = __get_pocket()
    for ligand_file in args.ligand:
        # use filename without extension as dataset name
        name = os.path.splitext(os.path.split(ligand_file)[1])[0]

        if args.verbose:
            print(f'reading {ligand_file}')
        try:
            ligand = next(pybel.readfile(args.ligand_format, ligand_file))
        except:
            raise IOError(f'Cannot read {ligand_file} file')

        ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
        pocket_coords, pocket_features = next(pocket_generator)

        centroid = ligand_coords.mean(axis=0)
        ligand_coords -= centroid
        pocket_coords -= centroid

        data = np.concatenate(
            (np.concatenate((ligand_coords, pocket_coords)),
             np.concatenate((ligand_features, pocket_features))),
            axis=1,
        )

        dataset = f.create_dataset(name, data=data, shape=data.shape,
                                   dtype='float32', compression='lzf')
        if affinities is not None:
            dataset.attrs['affinity'] = affinities.loc[name]
if args.verbose:
    print(f'\n\ncreated {args.output} with {num_ligands} structures')
