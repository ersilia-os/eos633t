import os
import sys
import concurrent.futures

os.environ.setdefault("GUNICORN_CMD_ARGS", "--timeout=3600")

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
from rdkit.Chem.Scaffolds import MurckoScaffold

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = os.path.dirname(os.path.abspath(__file__))
BLOCKS_LIST = os.path.join(
  ROOT, "..", "..", "checkpoints", "fragments_from_enamine.smi"
)
MODEL_DIR = os.path.abspath(
  os.path.join(ROOT, "..", "..", "checkpoints", "MODEL_DIR")
)

N_SAMPLES = 1000
PER_MOL_TIMEOUT = 300  # seconds; safety net for hanging decode calls


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


def decode_molecule(scaff, blocks_list_samp):
  # Model loaded per-call so TF releases memory on context exit — prevents OOM at large batch sizes.
  with load_model_from_directory(MODEL_DIR) as model:
    embs = model.encode([scaff])
    emb = np.array(embs[0])
    if emb.ndim == 1:
      emb = emb.reshape(1, -1)
    embeddings = np.repeat(emb, repeats=N_SAMPLES, axis=0)
    decoded = model.decode(embeddings, scaffolds=blocks_list_samp)
  return list(decoded) if decoded is not None else []


def main() -> None:
  supress_tensorflow_warnings()
  setup_logging()

  blocks_list = read_blocks()

  input_file = sys.argv[1]
  output_file = sys.argv[2]

  smiles_list = read_smiles(input_file=input_file)

  def empty_row():
    return [""] * N_SAMPLES

  R = [None] * len(smiles_list)

  for idx, smi in enumerate(tqdm(smiles_list, desc="Generating")):
    scaff = get_murcko_scaffold(smi)
    if scaff is None:
      print(f"[WARN] Invalid SMILES or empty scaffold at index {idx}: {smi!r}")
      R[idx] = empty_row()
      continue

    blocks_list_samp = random.sample(blocks_list, N_SAMPLES)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(decode_molecule, scaff, blocks_list_samp)
      try:
        result = future.result(timeout=PER_MOL_TIMEOUT)
      except concurrent.futures.TimeoutError:
        print(f"[WARN] decode timed out after {PER_MOL_TIMEOUT}s at index {idx}: {smi!r}")
        R[idx] = empty_row()
        continue
      except Exception as e:
        print(f"[ERROR] at index {idx} for {smi!r}: {type(e).__name__}: {e}")
        R[idx] = empty_row()
        continue

    result = (result + [""] * N_SAMPLES)[:N_SAMPLES]
    R[idx] = result

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
