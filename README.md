[![pipeline status](https://gitlab.com/cheminfIBB/pafnucy/badges/master/pipeline.svg)](https://gitlab.com/cheminfIBB/pafnucy/commits/master)

**Pafnucy [paphnusy]** is a 3D convolutional neural network that predicts binding affinity for protein-ligand complexes.
It was trained on the [PDBbind](http://pubs.acs.org/doi/abs/10.1021/acs.accounts.6b00491) database and tested on the [CASF](http://pubs.acs.org/doi/pdf/10.1021/ci500081m) "scoring power" benchmark.

The manuscript describing Pafnucy was published in *Bioinformatics* [DOI: 10.1093/bioinformatics/bty374](https://doi.org/10.1093/bioinformatics/bty374).

# Installation

In order to get Pafnucy, you need to clone this repo:

```
git clone https://gitlab.com/cheminfIBB/pafnucy
cd pafnucy
```

The easiest way to prepre the environment with all required packages is to use [conda](http://conda.io).
You can create environment with GPU-enabled version of tensorflow with:

```
conda env create -f environment_gpu.yml
```

Note that the environment contains [CUDA Toolkit](http://docs.nvidia.com/cuda) 8.0 and [cuDNN](https://developer.nvidia.com/cudnn) 5.1, so you do not need to install them on your system.

To create environment for CPU support only use:
```
conda env create -f environment_cpu.yml
```
Remember to activate your environment before running the scripts:
```
source activate pafnucy_env
```

Now you are ready to use Pafnucy:
```bash
python prepare.py -l ligand.mol2 -p pocket.mol2 -o data.hdf
python predict.py -i data.hdf -o predictions.csv
```


If get a segmentation fault with these commands, see [this discussion](https://gitlab.com/cheminfIBB/pafnucy/issues/6#note_130710151) for possible solution.

# Usage instructions

## Predict

This repository contains trained network, which can be used to score molecular complexes.
As input it takes 3D grids, with each grid point described with 19 atomic features.
You can create grids from molecular structures using functions defined in `tfbio.data` package or with `prepare.py` script.
Then you can load the network and make predictions with functions from `tfbio.net` package and Tensorflow API or with `predict.py` script.

### Prepare complexes

Save pockets and docked ligands into separate files and use `prepare.py` to create HDF file with atoms' coordinates and features.
By default, script expects mol2 files, but you can use any appropriate file format supported by [Open Babel](http://openbabel.org).
Note that Pafnucy uses protonation and partial charges to calculate features, so make sure that your files contain this information.

If you have a single file with a protein structure, use:

```bash
python prepare.py -l ligand1.mol2 ligand2.mol2 -p pocket.mol2 -o complexes.hdf
```

If you have different pocket for each ligand, list them all in the same order as ligands:

```bash
python prepare.py -l ligand1.mol2 ligand2.mol2 -p pocket1.mol2 pocket2.mol2 -o complexes.hdf
```

If you use different file format than mol2, specify it with the `--ligand_format` and/or `--pocket_format`:

```bash
python prepare.py -l ligand1.sdf ligand2.sdf --ligand_format sdf -p pocket.pdbqt --pocket_format pdbqt -o complexes.hdf
```

For more options, type:
```bash
python prepare.py --help
```

### Score complexes

When complexes are prepared, you can score them with the newtork:

```bash
python predict.py -i complexes.hdf -o predictions.csv
```

In case of our two ligands, `predictions.csv` will look like:
```
name,prediction
ligand1,<float>
ligand2,<float>

```

By default the script uses network from this repository.
If you want to use your own set of weights, use:

```bash
python predict.py -i complexes.hdf -n some/path/my_net -o predictions.csv
```

Note that network is defined with at least 3 files: `my_net.meta`, `my_net.index` and one or more `my_net.data-<number>-of-<number>` files, and all of them need to be present in `some/path` directory.
Also remember to specify `--grid_spacing` and `--max_dist`, if you used custom values of this parameters during training.
Also if you used different dataset than PDBbind v. 2016 or trained the model without using `training.py`, you should specify `--charge_scaler`.
If you did not scale the partial charges, use `--charge_scaler 1`.
If you trained your model with `training.py`, the partial charges were scaled by the standard deviation and the value was printed out:

```
---- FEATURES ----

atomic properties: ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal', 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'moltype', 'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']
charges: mean=-0.105995, sd=0.430907
use sd as scaling factor
```


## Train

In order to build and train new network use `train.py` script.
You can also do it manually using `tfbio.net` module and Tensorflow API.

First, prepare the structures and split your data into 3 subsets: training, validation and test set.
Save each subset as HDF file named `training_set.hdf` etc, with each complex as a dataset and its binding affinity saved in 'affinity' atribute for this dataset.

If you use the PDBbind dataset, you can prepare it with `pdbbind_data.ipynb` notebook and then use `split_dataset.py` script to split it into the 3 subsets.
The notebook assumes, that your files are organized as in PDBbind v. 2016, so if you use a different version you might need to change some paths.
The script `split_dataset.py` will by default use core set as test set, 1000 randomly selected molecules from refined set as validation set, and the rest of the data as training set.
You can control the size of validation set with `--size_val` attribute.

After creating the dataset you can use it to train the new network:

```bash
python training.py --input_dir path/to/dataset/
```

This script also allows you to specify network architecture, regularization and training parameters, and paths for output and logs.
To print all options, type:
```bash
python training.py --help
```





# Hyperparameters

## Default:
### Input:
* 1A grid
* 20A box (both ends included)
* 19 features:
    *  atom type: *B*, *C*, *N*, *O*, *P*, *S*, *Se*, *halogen*, and *metal* (one-hot or null, 9 columns)
    * hybridization (*hyb*, 1, 2, or 3)
    * connections with other heavy- and heteroatoms (*heavyvalence* and *heterovalence*)
    * additional properties defined with SMARTS patterns: *hydrophobic*, *aromatic*, *acceptor*, *donor*, and *ring* (binary)
    * partial charge (*partialcharge*, scaled by training set std)
    * ligand / protein (*moltype*, 1 for ligand, -1 for protein)

### Model
* architecture:
    * 3 convolutional layers with 64, 128, and 256 filters, each with 5A filter size and followed by max pooling with 2A patch size
    * 3 dense layers with 1000, 500, and 200 neurons
* initialization:
    * convolutional filters - weights draw from truncated normal with 0 mean and 0.001 std, all biases set to 0.1
    * dense layers - weights draw from truncated normal with 0 mean and 1/sqrt(fan_in) std, all biases set to 1.0
* regularization:
    * 0.5 dropout
    * 0.001 weight decay (L2)
* training:
    * 20 epochs
    * 20 samples per batch
    * Adam optimizer with 1e-5 learning rate
    * data augmentation (24 different orientations for each training case)
    * model evaluated after each epoch, saved only if validation error improved

## Other tested setups:
* 10 samples per batch - same error, slightly better correlation
* **5 samples per batch** - best performing model
* 0.2 dropout (keep_prob=0.8) / no dropout (keep_prob=1.0) - results for all sets are worse
* no weight decay - results are almost the same, but weights are much higher, especially for boron (B), which is present in only 166 compounds in the training set
* 0.01 weight decay - weights look much better, but results are worse (also for training set)


# Requirements
In order to use the provided model and run all scripts you need:
* Python 3.5
* tfbio 0.3
* tensorflow 1.0 (GPU-enabled version is **highly** recommended)
* openbabel 2.4
* numpy 1.12
* h5py 2.7
* matplotlib 2.0
* pandas 0.20
* scikit-learn 0.18.1
* seaborn 0.7


# References

[PDBbind database](http://pubs.acs.org/doi/abs/10.1021/acs.accounts.6b00491)

[CASF benchmark](http://pubs.acs.org/doi/pdf/10.1021/ci500081m)
