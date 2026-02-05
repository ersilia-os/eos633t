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
BLOCKS_LIST = os.path.join(
  ROOT, "..", "..", "checkpoints", "fragments_from_enamine.smi"
)

N_SAMPLES = 1000

import time


def call_with_retries(fn, *, tries=5, sleep_s=10, name="call"):
  last_err = None
  for i in range(tries):
    try:
      return fn()
    except RuntimeError as e:
      last_err = e
      print(
        f"[WARN] {name} failed (attempt {i + 1}/{tries}): {e}. Sleeping {sleep_s}s..."
      )
      time.sleep(sleep_s)
  raise last_err


def get_murcko_scaffold(smiles: str):
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return None
  scaff = MurckoScaffold.GetScaffoldForMol(mol)
  if scaff is None or scaff.GetNumAtoms() == 0:
    return None
  return Chem.MolToSmiles(scaff)


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
  model_directory = os.path.abspath(
    os.path.join(ROOT, "..", "..", "checkpoints", "MODEL_DIR")
  )
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

  ROOT = os.path.dirname(os.path.abspath(__file__))
  model_directory = os.path.abspath(
    os.path.join(ROOT, "..", "..", "checkpoints", "MODEL_DIR")
  )

  def empty_row():
    return [""] * N_SAMPLES

  def safe_scaffold(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
      return None
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    if scaff is None or scaff.GetNumAtoms() == 0:
      return None
    return Chem.MolToSmiles(scaff)

  R = []

  with load_model_from_directory(model_directory) as model:
    for idx, smi in enumerate(tqdm(smiles_list, desc="Generating")):
      try:
        if Chem.MolFromSmiles(smi) is None:
          print(f"[WARN] Invalid SMILES at index {idx}: {smi!r} -> writing empty row")
          R.append(empty_row())
          continue

        scaff = safe_scaffold(smi)
        if not scaff:
          print(
            f"[WARN] Empty/invalid scaffold at index {idx}: {smi!r} -> writing empty row"
          )
          R.append(empty_row())
          continue

        blocks_list_samp = random.sample(blocks_list, N_SAMPLES)
        row = call_with_retries(
          lambda: model.encode([scaff]), tries=6, sleep_s=5, name="encode"
        )

        emb = np.array(row)

        if emb.ndim == 1:
          emb = emb.reshape(1, -1)
        embeddings = np.repeat(emb, repeats=N_SAMPLES, axis=0)

        decoded = call_with_retries(
          lambda: model.decode(embeddings, scaffolds=blocks_list_samp),
          tries=6,
          sleep_s=5,
          name="decode",
        )
        if decoded is None:
          print(
            f"[WARN] decode returned None at index {idx}: {smi!r} -> writing empty row"
          )
          R.append(empty_row())
          continue

        decoded = list(decoded)
        if len(decoded) != N_SAMPLES:
          print(
            f"[WARN] decode length mismatch at index {idx}: got {len(decoded)} expected {N_SAMPLES} "
            f"-> padding/truncating"
          )
          decoded = (decoded + [""] * N_SAMPLES)[:N_SAMPLES]

        R.append(decoded)

      except Exception as e:
        print(
          f"[ERROR] Failed at index {idx} for SMILES {smi!r}: {type(e).__name__}: {e}"
        )
        R.append(empty_row())

  with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"smiles_{str(i).zfill(3)}" for i in range(N_SAMPLES)]
    writer.writerow(header)
    for row in R:
      if row is None:
        row = empty_row()
      row = (list(row) + [""] * N_SAMPLES)[:N_SAMPLES]
      writer.writerow(row)


if __name__ == "__main__":
  main()
