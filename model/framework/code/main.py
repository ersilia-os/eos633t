import os
import sys

from molecule_generation import load_model_from_directory
from molecule_generation.utils.cli_utils import (
  setup_logging,
  supress_tensorflow_warnings,
)
import csv
import random
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


def get_murcko_scaffold(smiles):
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


def scaffold_based_sampling(scaff, blocks_list, seed, target=N_SAMPLES, max_rounds=5):
  # Model loaded once per compound (not once for the whole run, and not once per round) —
  # TF releases memory when this context exits, avoiding the memory-accumulation OOM
  # previously fixed in c72c312. A single encode/decode round returns heavily duplicated
  # molecules (confirmed: median 75% unique, worst case 28% unique on this model's
  # 100-compound benchmark) — draw fresh non-overlapping fragment batches across multiple
  # rounds, deduping via canonical SMILES, until `target` unique molecules are collected.
  local_rng = random.Random(seed)
  pool = list(blocks_list)
  local_rng.shuffle(pool)

  seen = set()
  results = []
  idx = 0
  with load_model_from_directory(MODEL_DIR, seed=seed) as model:
    for _ in range(max_rounds):
      if len(results) >= target or idx >= len(pool):
        break
      batch = pool[idx: idx + target]
      idx += len(batch)

      embeddings = model.encode(batch)
      decoded = model.decode(embeddings, scaffolds=[scaff] * len(batch))
      for o in decoded:
        if not o:
          continue
        mol = Chem.MolFromSmiles(o)
        if mol is None:
          continue
        key = Chem.MolToSmiles(mol)
        if key in seen:
          continue
        seen.add(key)
        results.append(o)
        if len(results) >= target:
          break

  return results[:target]


def main() -> None:
  supress_tensorflow_warnings()
  setup_logging()

  blocks_list = read_blocks()

  input_file = sys.argv[1]
  output_file = sys.argv[2]

  smiles_list = read_smiles(input_file=input_file)

  # Our own RNG instance, independent of the global `random` module — ModelWrapper
  # resets the global module's seed to a fixed value on every model load, which would
  # otherwise make this sampling collapse to the same draw after the first compound.
  rng = random.Random()

  def empty_row():
    return [""] * N_SAMPLES

  R = [None] * len(smiles_list)

  for idx, smi in enumerate(tqdm(smiles_list, desc="Generating")):
    scaff = get_murcko_scaffold(smi)
    if scaff is None:
      print(f"[WARN] Invalid SMILES or empty scaffold at index {idx}: {smi!r}")
      R[idx] = empty_row()
      continue

    max_retries = 10
    result = []
    for attempt in range(max_retries):
      seed = rng.randint(1, 99999)
      try:
        result = scaffold_based_sampling(scaff, blocks_list, seed)
      except Exception as e:
        print(
          f"[ERROR] at index {idx} for {smi!r} (attempt {attempt + 1}/{max_retries}): "
          f"{type(e).__name__}: {e}"
        )
        result = []
      if result:
        break
      if attempt < max_retries - 1:
        print(
          f"[WARN] Empty result at index {idx} for {smi!r}, "
          f"retrying (attempt {attempt + 1}/{max_retries})"
        )

    result = (result + [""] * N_SAMPLES)[:N_SAMPLES]
    R[idx] = result

  with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"smi_{str(i).zfill(3)}" for i in range(N_SAMPLES)]
    writer.writerow(header)
    for row in R:
      if row is None:
        row = empty_row()
      row = (list(row) + [""] * N_SAMPLES)[:N_SAMPLES]
      writer.writerow(row)


if __name__ == "__main__":
  main()
