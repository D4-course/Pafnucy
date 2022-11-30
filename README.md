# Pafnucy
**Pafnucy [paphnusy]** is a 3D convolutional neural network that predicts binding affinity for protein-ligand complexes.
It was trained on the [PDBbind](http://pubs.acs.org/doi/abs/10.1021/acs.accounts.6b00491) database and tested on the [CASF](http://pubs.acs.org/doi/pdf/10.1021/ci500081m) "scoring power" benchmark.

The manuscript describing Pafnucy was published in Bioinformatics DOI: 10.1093/bioinformatics/bty374.
## System Requirements: 
- Docker

## Installation Instructions:
- Open a terminal and clone the repo
```
git clone https://github.com/D4-course/Pafnucy.git
```
- Download [trained models](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/savitha_viswanadh_research_iiit_ac_in/EZRehdh5o0FIpLbiPAb97r8By7_YZaRP7oQZufSOqz3FnA?e=ZuT8L4) and load them into ```Pafnucy/results``` folder in the parent repo. 
- Execute ```run.sh``` that will build the docker image, run a container where the frontend and backend will be executed automatically
```
./run.sh
or
sh run.sh
```
- Navigate to the network or external URL that will be displayed on the terminal
```

You can now view your Streamlit app in your browser.

  Network URL: http://10.42.0.88:8501
  External URL: http://10.1.34.46:8501
```
## Website Instructions
- Firstly, select a ligand file (```.mol2```) and pocket file (```.mol2```) of interest and upload it. (Sample ```ligand.mol2```, ```pocket.mol2``` files attached in the repo)
- Click on ```predict``` to get a list of protien binding affinities of the uploaded ligands-pocket combination.
