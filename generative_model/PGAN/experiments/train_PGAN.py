import pandas as pd
import numpy as np
import glob
import os
from os.path import join, dirname
import json

def create_json_file(tissue):
    data_tissue = {
    "pathDB": "../../data/example_embeddings.h5ad",
    "config": {
        "maxIterAtScale": [
        48000,
        96000,
        96000,
        96000,
        96000,
        96000,
        1000000
        ]
    }
    }
    file_path = f'../config/config.json'

    with open(file_path, "w") as json_file:
        json.dump(data_tissue, json_file, indent=4)

os.chdir('..')
cmd = "python train.py PGAN -c ./config/config.json -n reecap --dir ../data/output"
os.system(cmd)
