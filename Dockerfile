FROM continuumio/miniconda3

WORKDIR /usr/src/app

# Create the environment:
COPY ./pafnucy/environment_cpu.yml environment.yml
RUN conda env create -f environment.yml

COPY ./pafnucy ./p

SHELL ["conda", "run", "-n", "pafnucy_env", "/bin/bash", "-c"]

WORKDIR p

#ENTRYPOINT ["conda", "run", "-n", "pafnucy_env", "python" , "prepare.py" , "-l" , "3ui7_ligand.mol2" , "-p" , "3ui7_pocket.mol2" , "-o" , "data.hdf"]

