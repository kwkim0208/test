import os
import glob
import warnings
import time
import logging
import numpy as np
import math
import time
import argparse
from multiprocessing import Process, Queue
import utils
import pandas as pd

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from moses.metrics import SA
from rdkit.Chem.Descriptors import qed
from tqdm import tqdm
import modification_utils

import numpy as np
from keras.models import model_from_json 
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance

def do_chemical_transformation(input_smiles_list, reaction_rule_info):
    total_smiles = {}

    for i in tqdm(range(0, len(input_smiles_list)),mininterval=1):
        input_smiles = input_smiles_list[i]
        input_mol = Chem.MolFromSmiles(input_smiles)
        input_mol_atom_num = len(input_mol.GetAtoms())
        mol_cut_off = int(input_mol_atom_num*0.2)
        
        key_list = list(reaction_rule_info.keys())        
        
        for j in tqdm(range(0, len(key_list)),mininterval=1):
            each_rule = key_list[j]
            pattern_smiles = each_rule[0]
            replace_smiles = each_rule[1]
            template_smiles = each_rule[2]
            reaction_type = each_rule[3]
            
            pattern_mol = Chem.MolFromSmiles(pattern_smiles)
            replace_mol = Chem.MolFromSmiles(replace_smiles)
            
            if pattern_mol != None and replace_mol != None:
                pattern_mol_atom_num = len(pattern_mol.GetAtoms())
                replace_mol_atom_num = len(replace_mol.GetAtoms())
                if reaction_type == 'ADD':
                    if pattern_mol_atom_num <= mol_cut_off:
                        try:
                            results = modification_utils.add_substructures(input_mol, replace_mol, pattern_mol)
                        except:
                            continue
                        for smiles in results:
                            total_smiles[smiles] = 1
                            
                if reaction_type == 'REMOVE':
                    if pattern_mol_atom_num <= mol_cut_off:
                        s = time.time()
                        try:
                            results = modification_utils.remove_substructures(input_mol, pattern_mol)
                        except:
                            continue
                        for smiles in results:
                            total_smiles[smiles] = 1
                            
                if reaction_type == 'REPLACE':
                    if abs(pattern_mol_atom_num-replace_mol_atom_num) <= mol_cut_off:
                        template_mol = Chem.MolFromSmiles(template_smiles)
                        if template_mol != None:
                            try:
                                results = modification_utils.replace_substructures(input_mol, pattern_mol, replace_mol, template_mol)
                            except:
                                continue

                            for smiles in results:
                                total_smiles[smiles] = 1
    return total_smiles

def run_filtering(input_smiles_list, total_smiles, threshold):
    target_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'H']
    max_weight = 800
    
    input_mol_list = []
    generative_mol_list = []
    generative_smiles_list = []
    final_mol_list = []
    final_mol_sa_list = []
    
    for smiles in input_smiles_list:
        input_mol = Chem.MolFromSmiles(smiles)
        input_mol_list.append(input_mol)
    
    for smiles in total_smiles:
        tmp_mol = Chem.MolFromSmiles(smiles)
        generative_mol_list.append(tmp_mol)
        generative_smiles_list.append(smiles)

    for i in tqdm(range(0, len(input_mol_list)),mininterval=1):
        try:
            each_input_mol = input_mol_list[i]
            fp1 = AllChem.GetMorganFingerprint(each_input_mol,2)
        except:
            continue

        for j in tqdm(range(0, len(generative_mol_list)),mininterval=1):
            try:
                each_tmp_mol = generative_mol_list[j]
                fp2 = AllChem.GetMorganFingerprint(each_tmp_mol,2)
            except:
                continue
            
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            
            if sim < threshold:
                continue
                
            if sim == 1:
                continue
                
            if Descriptors.ExactMolWt(each_tmp_mol) > max_weight:
                continue 

            flag = True
            for each_atom in each_tmp_mol.GetAtoms():
                if each_atom.GetSymbol() not in target_atoms:
                    flag = False
            
            if flag == False:
                continue
            
            smiles = generative_smiles_list[j]
            final_mol_list.append(smiles)
    return final_mol_list

def run_transformation(input_smiles_list, rule_filename):
    transformation_pattern_info = utils.read_chemical_transformation_rules(rule_filename)
    total_smiles = do_chemical_transformation(input_smiles_list, transformation_pattern_info)
    candidate_smiles = check_feasibility(input_smiles_list, total_smiles)
    return candidate_smiles

def check_feasibility(input_smiles_list, total_smiles):
    reaction_model = './data/reaction_model.json'
    reaction_model_weight = './data/reaction_model.h5'
    
    compound_model = './data/compound_model.json'
    compound_model_weight = './data/compound_model.h5'
    
    json_file = open(reaction_model, "r")
    reaction_loaded_model_json = json_file.read() 
    json_file.close()
    reaction_loaded_encoder_model = model_from_json(reaction_loaded_model_json)
    reaction_loaded_encoder_model.load_weights(reaction_model_weight)
    
    json_file = open(compound_model, "r")
    compound_loaded_model_json = json_file.read() 
    json_file.close()
    compound_loaded_encoder_model = model_from_json(compound_loaded_model_json)
    compound_loaded_encoder_model.load_weights(compound_model_weight)

    reaction_feature, compound_features, smiles_candidates = utils.calc_reaction_features(input_smiles_list, total_smiles)
    
    compound_results = compound_loaded_encoder_model.predict(compound_features)
    compound_results = np.where(compound_results > 0.5, 1, 0)

    feasible_compound_smiles = []
    for i in tqdm(range(len(compound_features))):
        original_feat = compound_features[i]
        pred_feat = compound_results[i]
        smiles = smiles_candidates[i]
        
        dist = np.linalg.norm(original_feat-pred_feat)    
        cos_sim = dot(original_feat, pred_feat)/(norm(original_feat)*norm(pred_feat))
        if cos_sim > 0.80:
            feasible_compound_smiles.append(smiles)

    reaction_feature, compound_features, smiles_candidates = utils.calc_reaction_features(input_smiles_list, feasible_compound_smiles)
    reaction_results = reaction_loaded_encoder_model.predict(reaction_feature)
    reaction_results = np.where(reaction_results > 0.5, 1, 0)

    feasible_compound_smiles = []
    for i in tqdm(range(len(reaction_feature))):
        original_feat = reaction_feature[i]
        pred_feat = reaction_results[i]
        smiles = smiles_candidates[i]
        dist = np.linalg.norm(original_feat-pred_feat)    
        cos_sim = dot(original_feat, pred_feat)/(norm(original_feat)*norm(pred_feat))
        
        if cos_sim > 0.60:
            feasible_compound_smiles.append(smiles)

    return feasible_compound_smiles

def calculate_property(top_n, input_smiles, smiles_list):
    input_mol = Chem.MolFromSmiles(input_smiles)
    fp1 = AllChem.GetMorganFingerprint(input_mol,2)
    
    idx_list = []
    cmp_cnt = 1
    cmp_info = {}
    for each_smiles in smiles_list:
        cmp_id = 'ID %s'%(cmp_cnt)
        cmp_info[cmp_id]={}
        each_tmp_mol = Chem.MolFromSmiles(each_smiles)
        fp2 = AllChem.GetMorganFingerprint(each_tmp_mol,2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        
        cmp_info[cmp_id]['Smiles'] = each_smiles
        cmp_info[cmp_id]['Similarity'] = sim
        cmp_cnt+=1
        
    df = pd.DataFrame.from_dict(cmp_info)
    df = df.T
    df = df.astype({"Similarity": float})
    df = df.nlargest(top_n, 'Similarity')
    
    results = {}
    cmp_cnt = 1
    for idx, each_df in df.iterrows():
        cmp_id = 'CID %s'%(cmp_cnt)
        smiles = each_df['Smiles']
        results[cmp_id] = smiles
        cmp_cnt+=1
    return results

def main(smiles):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    
    warnings.filterwarnings('ignore')
    start = time.time()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    chemical_transformation_rule_file = './data/Chemical_transformation_patterns.txt'
    similarity_threshold = 0.7
    
    top_n = 100
    input_smiles_list = [smiles]
    total_smiles = run_transformation(input_smiles_list, chemical_transformation_rule_file)
    final_smiles_list = run_filtering(input_smiles_list, total_smiles, similarity_threshold)
        
    df = calculate_property(top_n, input_smiles_list[0], final_smiles_list)
    
    logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))
    
    return df

    
    
if __name__ == '__main__':
    main()
    




