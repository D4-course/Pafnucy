'''
Api for Deeppocket using Fastapi
    /api/v1/ - api version 1
    /api/v1/rank/protein/{protein}/num_pockets/{num_pockets} - rank pockets
    /api/v1/segment?protein=protein.pdb&num_pockets=10 - segment pockets
'''
from fastapi import FastAPI, File, UploadFile
import shutil
import os
import uvicorn
BASE_API = "/api/v1"


# function to predict ranks
app = FastAPI()

# post api for receivng two file
@app.post(BASE_API + "/score/")
async def score(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    '''
    predict score of protein and ligand complex'''
    # write file to dist as ligand.mol2
    with open("ligand.mol2", "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    # write file to dist as protein.mol2
    with open("protein.mol2", "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)
    # run the following commands using os
    # python prepare.py -l ligand.mol2 -p pocket.mol2 -o data.hdf
    # python predict.py -i data.hdf -o predictions.csv
    os.system("conda run -n pafnucy_env python prepare.py -l ligand.mol2 -p protein.mol2 -o data.hdf")
    os.system("conda run -n pafnucy_env python predict.py -i data.hdf -o predictions.csv")
    # read the predictions.csv file and return the score
    binding_affinity_score = 0
    with open("predictions.csv", "r", encoding="utf-8") as file:
        binding_affinity_score = file.read().split(",")[1]
    print(binding_affinity_score)
    os.remove("ligand.mol2")
    os.remove("protein.mol2")
    os.remove("data.hdf")
    #os.remove("predictions.csv")
    return {"score": binding_affinity_score}

if __name__ == "__main__":
    uvicorn.run(app, host = "localhost" ,port=8000)
