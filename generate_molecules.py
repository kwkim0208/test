import random
import subprocess
import tempfile
import os
import pandas as pd
import run_reaction_api
import json

def prediction(smiles):
    results = run_reaction_api.main(smiles)
    json_data = json.dumps(results)
    return json_data

smiles = 'Cc1cn2c(n1)CCC(NC(=O)Cc1c(C)noc1Cl)C2'
json_data = prediction(smiles)
print (json_data)
