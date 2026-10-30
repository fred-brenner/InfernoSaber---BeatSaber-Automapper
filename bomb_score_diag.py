"""Diagnostic: are Arm E bomb scores informative? (teacher-forced, post-mask)

Loads a few validation songs from the feasible dataset, runs the trained
hazards model, and summarizes where reference bombs rank among eligible cells.
"""
import numpy as np

from training.pipeline_v2.artifacts import load_dataset
from training.pipeline_v2.calibration import _validation_songs, teacher_forced_outputs
from training.pipeline_v2.runner import load_stage_artifact

DATASET = "../Data/training/pipeline_v2/full-2026-10-27-feasible/dataset"
ARTIFACT = "../Data/training/pipeline_v2/runs/feasible-20261027-w50-c50/hazards"
N_SONGS = 6

dataset = load_dataset(DATASET)
model, _ = load_stage_artifact(ARTIFACT, "hazards", dataset)
options = dataset.manifest.get("data_settings", {})
predictor = lambda inputs: model(inputs, training=False)

songs = 0
pos_scores, neg_scores, eligible_all, pos_ranks = [], [], [], []
ref_per_song = []
for song in _validation_songs(dataset, "hazards"):
    ex = song.examples["hazards"]
    if not len(ex.times):
        continue
    outputs = teacher_forced_outputs(song, "hazards", predictor, dataset.vocabularies, options)
    bombs = np.asarray(outputs["bombs"])          # (N,12) post-mask sigmoids
    targets = np.asarray(ex.targets["bombs"])     # (N,12)
    weights = np.asarray(ex.weights["bombs"])
    feasible = np.asarray(ex.inputs["feasible_bombs"]).astype(bool)
    for i in np.flatnonzero(np.asarray(ex.times) >= -1):
        elig = feasible[i] & (weights[i] > 0)
        if not elig.any():
            continue
        scores = bombs[i][elig]
        eligible_all.append(scores)
        for c in np.flatnonzero(elig):
            if targets[i, c] > 0:
                pos_scores.append(bombs[i, c])
                pos_ranks.append(float((scores < bombs[i, c]).mean()))
            else:
                neg_scores.append(bombs[i, c])
    ref_per_song.append(int((targets > 0).sum()))
    songs += 1
    if songs >= N_SONGS:
        break

eligible_all = np.concatenate(eligible_all)
pos_scores = np.asarray(pos_scores)
neg_scores = np.asarray(neg_scores)
print(f"songs={songs} eligible_cells={len(eligible_all)} ref_bombs={ref_per_song}")
print(f"eligible score percentiles: {np.percentile(eligible_all, [50, 90, 99, 99.9]).round(4)}")
print(f"POSITIVE cells n={len(pos_scores)} mean={pos_scores.mean():.4f} "
      f"p50={np.percentile(pos_scores, 50):.4f} p90={np.percentile(pos_scores, 90):.4f}")
print(f"negative cells n={len(neg_scores)} mean={neg_scores.mean():.4f} "
      f"p50={np.percentile(neg_scores, 50):.4f} p90={np.percentile(neg_scores, 90):.4f}")
print(f"positive mean rank among query's eligible cells: {np.mean(pos_ranks):.3f} "
      f"(0.5 = uninformative; >0.9 = strong ranking)")
print(f"share of positives with score >= 0.5: {(pos_scores >= 0.5).mean():.3f}")
print(f"share of negatives with score >= 0.5: {(neg_scores >= 0.5).mean():.4f}")
