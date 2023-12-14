#!/usr/bin/env python3
import os
import sys 

from molecule_generation import load_model_from_directory
from molecule_generation.utils.cli_utils import (
    setup_logging,
    supress_tensorflow_warnings,
)
import csv
import random
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = os.path.dirname(os.path.abspath(__file__))
BLOCKS_LIST = os.path.join(ROOT, "..", "..", "checkpoints", "fragments_from_enamine.smi")

N_SAMPLES = 1000


def get_murcko_scaffold(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles


def read_blocks():
    blocks_list = []
    with open(BLOCKS_LIST, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for r in reader:
            blocks_list += [r[0]]
    return blocks_list

def read_smiles(input_file):
    smiles = []
    with open(input_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            smiles += [r[0]]
    print("These are the SMILES: ", smiles)
    return smiles

def tanimoto_calc(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
    s = round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)
    return s


def scaffold_based_sampling(query_scaffold, blocks_list):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    model_directory = os.path.abspath(os.path.join(ROOT,"..","..","checkpoints","MODEL_DIR"))
    with load_model_from_directory(model_directory) as model:
        row = model.encode([query_scaffold])
        R = []
        for _ in range(len(blocks_list)):
            R += row
        embeddings = np.array(R)
        print(embeddings.shape)
        decoded = model.decode(embeddings, scaffolds=blocks_list)
    return decoded


def main() -> None:
    supress_tensorflow_warnings()
    setup_logging()

    blocks_list = read_blocks()
	
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    smiles_list = read_smiles(input_file=input_file)
    scaff_list = [get_murcko_scaffold(smi) for smi in smiles_list]

    R = []
    for smi in tqdm(scaff_list):
        blocks_list_samp = random.sample(blocks_list, N_SAMPLES)
        decoded = scaffold_based_sampling(smi, blocks_list_samp)
        R += [decoded]
    
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        header = ["cpd_{0}".format(i) for i in range(N_SAMPLES)]
        writer.writerow(header)
        for r in R:
            writer.writerow(r)


if __name__ == "__main__":
    main()