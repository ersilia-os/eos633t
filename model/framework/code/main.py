import os
import sys

from molecule_generation import load_model_from_directory
from molecule_generation.utils.cli_utils import (
  setup_logging,
  supress_tensorflow_warnings,
)
import csv
import random
import time
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = os.path.dirname(os.path.abspath(__file__))
BLOCKS_LIST = os.path.join(
  ROOT, "..", "..", "checkpoints", "fragments_from_enamine.smi"
)

N_SAMPLES = 1000


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


def encode_scaffolds(model, scaffolds):
  """Encode with hierarchical fallback: batches of 100 -> 10 -> 1.
  Returns a list aligned with scaffolds; None at positions that failed."""
  results = [None] * len(scaffolds)

  def _encode(local_indices):
    if not local_indices:
      return
    chunk = [scaffolds[i] for i in local_indices]
    n = len(local_indices)
    try:
      embs = call_with_retries(
        lambda: model.encode(chunk), tries=6, sleep_s=5, name=f"encode(n={n})"
      )
      for j, i in enumerate(local_indices):
        results[i] = np.array(embs[j])
    except Exception as e:
      if n == 1:
        print(f"[WARN] encode failed for scaffold {scaffolds[local_indices[0]]!r}: {e}")
        return
      next_size = 10 if n > 10 else 1
      print(f"[WARN] encode(n={n}) failed ({e}), retrying in batches of {next_size}.")
      for start in range(0, n, next_size):
        _encode(local_indices[start:start + next_size])

  for start in range(0, len(scaffolds), 100):
    _encode(list(range(start, min(start + 100, len(scaffolds)))))

  return results


def main() -> None:
  supress_tensorflow_warnings()
  setup_logging()

  blocks_list = read_blocks()

  input_file = sys.argv[1]
  output_file = sys.argv[2]

  smiles_list = read_smiles(input_file=input_file)

  model_directory = os.path.abspath(
    os.path.join(ROOT, "..", "..", "checkpoints", "MODEL_DIR")
  )

  def empty_row():
    return [""] * N_SAMPLES

  # Pre-compute scaffolds before opening the model context
  scaffolds = []
  for idx, smi in enumerate(smiles_list):
    scaff = get_murcko_scaffold(smi)
    if scaff is None:
      print(f"[WARN] Invalid SMILES or empty scaffold at index {idx}: {smi!r}")
    scaffolds.append(scaff)

  valid_indices = [i for i, s in enumerate(scaffolds) if s is not None]
  valid_scaffolds = [scaffolds[i] for i in valid_indices]

  R = [None] * len(smiles_list)

  with load_model_from_directory(model_directory) as model:
    all_embeddings = encode_scaffolds(model, valid_scaffolds) if valid_scaffolds else []

    for i, idx in enumerate(tqdm(valid_indices, desc="Generating")):
      try:
        emb = all_embeddings[i]
        if emb is None:
          print(f"[WARN] No embedding for index {idx}: {smiles_list[idx]!r} -> writing empty row")
          R[idx] = empty_row()
          continue
        if emb.ndim == 1:
          emb = emb.reshape(1, -1)
        embeddings = np.repeat(emb, repeats=N_SAMPLES, axis=0)
        blocks_list_samp = random.sample(blocks_list, N_SAMPLES)

        decoded = call_with_retries(
          lambda: model.decode(embeddings, scaffolds=blocks_list_samp),
          tries=6, sleep_s=5, name="decode",
        )

        if decoded is None:
          print(f"[WARN] decode returned None at index {idx}: {smiles_list[idx]!r} -> writing empty row")
          R[idx] = empty_row()
          continue

        decoded = list(decoded)
        if len(decoded) != N_SAMPLES:
          print(
            f"[WARN] decode length mismatch at index {idx}: got {len(decoded)} "
            f"expected {N_SAMPLES} -> padding/truncating"
          )
          decoded = (decoded + [""] * N_SAMPLES)[:N_SAMPLES]

        R[idx] = decoded

      except Exception as e:
        print(f"[ERROR] Failed at index {idx} for SMILES {smiles_list[idx]!r}: {type(e).__name__}: {e}")
        R[idx] = empty_row()

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
