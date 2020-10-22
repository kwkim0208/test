from rdkit.Chem import rdFMCS
from rdkit import Chem
from rdkit.Chem import AllChem
import random
import copy
import sys
import time

def get_atom_object_from_idx(mol, atom_idx_list):
    atom_obj_list = []
    for each_idx in atom_idx_list:
        atom_obj_list.append(mol.GetAtomWithIdx(each_idx))
    return atom_obj_list

def get_neighbor_atoms(mol, idx, max_len=1):
    neighbors = mol.GetAtomWithIdx(idx).GetNeighbors()
    idx_list = []
    for each_neighbor in neighbors:
        idx_list.append(each_neighbor.GetIdx())
    
    tmp_idx_list = copy.deepcopy(idx_list)
    for i in range(max_len-1):
        for each_idx in tmp_idx_list:
            neighbors = mol.GetAtomWithIdx(each_idx).GetNeighbors()
            for each_neighbor in neighbors:
                idx_list.append(each_neighbor.GetIdx())
                
        idx_list = list(set(idx_list))
        tmp_idx_list = copy.deepcopy(idx_list)
    
    return idx_list

def get_atoms_from_bond(mol, bond_idx):
    begin_atom = None
    end_atom = None
    for bond in mol.GetBonds():
        if bond.GetIdx() == bond_idx:
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
    return begin_atom, end_atom

def get_bonds_from_atom_list(mol, atom_list):
    bonds = []
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        if begin_atom in atom_list and end_atom in atom_list:
            bonds.append(bond.GetIdx())
    return bonds

def get_submol_from_atom_list(mol, atom_list):
    if len(atom_list) == 1:
        atom_idx = atom_list[0]
        symbol = get_symbol_from_atom(mol, atom_idx)
        tmp_submol = Chem.MolFromSmiles(symbol)
    else:
        bonds = get_bonds_from_atom_list(mol, atom_list)
        tmp_submol = Chem.PathToSubmol(mol, bonds)
    return tmp_submol

def get_symbol_from_atom(mol, atom_idx):
    symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
    return symbol

def check_fragment_from_mol(mol):
    smiles = Chem.MolToSmiles(mol)
    flag = check_fragment_from_smiles(smiles)
    return flag

def check_fragment_from_smiles(smiles):
    flag = True
    if '.' in smiles:
        flag = False
    return flag

def get_MCS_atom_list(mol1, mol2):
    mols = [mol1, mol2]
    res = rdFMCS.FindMCS(mols)
    mcs_mol = Chem.MolFromSmarts(res.smartsString)
    
    mol1_atom_list = list(mol1.GetSubstructMatches(mcs_mol)[0])
    mol2_atom_list = list(mol2.GetSubstructMatches(mcs_mol)[0])
    smiles = Chem.MolToSmiles(mcs_mol)
    return mol1_atom_list, mol2_atom_list

def get_substructure_info(mol, pattern):
    substructure_candidate_atom_list = []
    for each_substr in mol.GetSubstructMatches(pattern):
        substructure_candidate_atom_list.append(each_substr)
    return substructure_candidate_atom_list

def get_mapping_atoms_from_substr(mol, atom_list):
    mapping_atom_list = []
    for idx in atom_list:
        neighbor_atoms = get_neighbor_atoms(mol, idx, max_len=1)
        extra_neighbor_atoms = set(neighbor_atoms).difference(set(atom_list))
        if len(extra_neighbor_atoms) == 1:
            mapping_atom_list.append(idx)
    return mapping_atom_list

def remove_bond_between_two_atom_list(mol, atom_set_list):
    edit_mol = Chem.EditableMol(mol)
    for atom_idx1, atom_idx2 in atom_set_list:
        edit_mol.RemoveBond(atom_idx1, atom_idx2)
    result_mol = edit_mol.GetMol()
    return result_mol

def remove_substructures(mol, pattern_mol):
    substructure_list = get_substructure_info(mol, pattern_mol)
    candidate_structures = []
    for each_structure_atom_list in substructure_list:
        temp_mol = copy.deepcopy(mol)
        edit_mol = Chem.EditableMol(temp_mol)
        
        for each_atom_idx in sorted(each_structure_atom_list, reverse=True):
            edit_mol.RemoveAtom(each_atom_idx)
        temp_mol = edit_mol.GetMol()
        smiles = Chem.MolToSmiles(temp_mol)
        if '.' in smiles:
            continue        
      
        if temp_mol != None:
            candidate_structures.append(smiles)
    return candidate_structures

def add_substructures(mol, pattern_mol, repl_mol):
    substructure_list = get_substructure_info(mol, pattern_mol)
    candidate_structures = []
    for each_structure_atom_list in substructure_list:
        if len(each_structure_atom_list) != 1:
            continue
            
        target_atom = each_structure_atom_list[0]
        temp_mol = copy.deepcopy(mol)
        temp_repl_mol = copy.deepcopy(repl_mol)
        mol_atoms = [each_atom.GetIdx() for each_atom in temp_mol.GetAtoms()]

        edit_temp_mol = Chem.EditableMol(temp_mol)
        combined_mol = Chem.CombineMols(mol, repl_mol)
        combined_edit_mol = Chem.EditableMol(combined_mol)
        
        candidate_atoms = []
        for each_atom in combined_mol.GetAtoms():
            if each_atom.GetIdx() not in mol_atoms:
                candidate_atoms.append(each_atom.GetIdx())

        for each_candidate_atom_idx in candidate_atoms:
            for bond_type in [Chem.rdchem.BondType.SINGLE]:
                temp_combined_mol = copy.deepcopy(combined_mol)
                temp_combined_edit_mol = Chem.EditableMol(temp_combined_mol)
                temp_combined_edit_mol.AddBond(each_candidate_atom_idx, target_atom, order=bond_type)
                candidate = temp_combined_edit_mol.GetMol()
                
                if candidate != None:
                    smiles = Chem.MolToSmiles(candidate)
                    candidate_structures.append(smiles)       
    return candidate_structures

def replace_single_atom(mol, pattern_mol, repl_mol):
    candidate_structure_info = {}
    rms = AllChem.ReplaceSubstructs(mol,pattern_mol,repl_mol)
    for each_structure in rms:
        smiles = Chem.MolToSmiles(each_structure) # isomericSmiles=False
        candidate_structure_info[smiles]=1
    candidate_structures = list(candidate_structure_info.keys())
    return candidate_structures

def replace_substructures(mol, pattern_mol, repl_mol, reference_mol):
    if pattern_mol.GetNumAtoms() == 1 and repl_mol.GetNumAtoms() == 1:
        candidate_structures = []
        candidate_structures = replace_single_atom(mol, pattern_mol, repl_mol)
        return candidate_structures
    else:
        substructure_list = get_substructure_info(mol, pattern_mol)
        candidate_structures = []
        for each_structure_atom_list in substructure_list:
            tmp_mol = copy.deepcopy(mol)
            mol_atoms = [each_atom.GetIdx() for each_atom in tmp_mol.GetAtoms()]
            mapping_candidate_atoms = []
            
            for each_atom in each_structure_atom_list:
                tmp_mapping_atoms = get_neighbor_atoms(mol, each_atom, max_len=1)
                tmp_target_atoms = list(set(tmp_mapping_atoms).difference(set(each_structure_atom_list)))
                if len(tmp_target_atoms) == 1:
                    mapping_candidate_atoms.append([each_atom, tmp_target_atoms[0]])

            if len(mapping_candidate_atoms) != 1:
                continue

            rm_atom_idx = mapping_candidate_atoms[0][0]
            link_atom_idx = mapping_candidate_atoms[0][1]
            
            rm_bond_tmp_mol = remove_bond_between_two_atom_list(tmp_mol, [[rm_atom_idx, link_atom_idx]])
            smiles = Chem.MolToSmiles(rm_bond_tmp_mol)

            target_atom = link_atom_idx
            edit_temp_mol = Chem.EditableMol(rm_bond_tmp_mol)
            combined_mol = Chem.CombineMols(rm_bond_tmp_mol, repl_mol)
            combined_edit_mol = Chem.EditableMol(combined_mol)
            candidate_atoms = []
            for each_atom in combined_mol.GetAtoms():
                if each_atom.GetIdx() not in mol_atoms:
                    candidate_atoms.append(each_atom.GetIdx())

            for each_candidate_atom_idx in candidate_atoms:
                for bond_type in [Chem.rdchem.BondType.SINGLE]:
                    temp_combined_mol = copy.deepcopy(combined_mol)
                    temp_combined_edit_mol = Chem.EditableMol(temp_combined_mol)
                    temp_combined_edit_mol.AddBond(each_candidate_atom_idx, target_atom, order=bond_type)
                    temp_mol = temp_combined_edit_mol.GetMol()
                    
                    smiles = Chem.MolToSmiles(temp_mol)
                    if '.' in smiles:
                        continue

                    search_results = temp_mol.GetSubstructMatches(reference_mol)            
                    if len(search_results)>0:
                        if temp_mol != None:
                            candidate_structures.append(smiles)

        return candidate_structures