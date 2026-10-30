import numpy as np
from training.pipeline_v2.artifacts import load_dataset
from training.pipeline_v2.calibration import _validation_songs, _tolerant_match, _reference_state
from training.pipeline_v2.runner import load_stage_artifact
from training.pipeline_v2.hazard_sections import candidates, SectionDecoder
from training.pipeline_v2.contracts import build_inputs

dataset = load_dataset("../Data/training/pipeline_v2/full-2026-10-27-feasible/dataset", lazy=True)
options = dataset.manifest["data_settings"]
model, _ = load_stage_artifact("../Data/training/pipeline_v2/runs/feasible-20261027-w50-c50/hazards", "hazards", dataset)
import tensorflow as tf
predictor = tf.function(lambda i: model(i, training=False), reduce_retracing=True)
vocab = dataset.vocabularies
targets = {"6ad84b9f153c8c669bb05018", "6f73d981e30c3e252305f578",
           "68e8b64174e87891bfe3f4cd", "610e52fb5ddbaca38e95fe89"}
songs = [s for s in _validation_songs(dataset, "hazards") if s.base.song_id in targets]

def replay(song, qthr, strip_own):
    state = _reference_state(song, vocab, options)
    queries = candidates(song.base.metadata["duration"], song.base.metadata["bpm"], options)
    decoder = SectionDecoder(state, song.base.metadata["duration"], vocab, options)
    for query in queries:
        cand = dict(query)
        cand["_feasible_wall"] = (decoder.feasible_wall_classes(query) if query["kind"] == "wall"
                                  else np.r_[True, np.zeros(len(vocab["walls"]) - 1, bool)])
        cand["_feasible_bombs"] = decoder.eligible_bomb_cells(float(query["time"]))
        out = predictor(build_inputs("hazards", song, state, cand, decoder.history, None, options))
        wall, bombs = np.asarray(out["wall"])[0], np.asarray(out["bombs"])[0]
        choice, fires = int(wall.argmax()), np.zeros(12, bool)
        if query["kind"] == "wall":
            feas = decoder.feasible_wall_classes(query)
            if choice and not feas[choice]:
                choice = 0
        else:
            elig = decoder.eligible_bomb_cells(float(query["time"]))
            mx = bombs[elig].max() if elig.any() else 0.
            fires[np.argmax(np.where(elig, bombs, -1.))] = mx >= qthr
        decoder.step(query, choice, fires)
        if strip_own and query["kind"] == "bomb" and decoder.history:
            decoder.history[-1]["bombs"] = np.zeros(12)
    return decoder.finish()


for strip in (False, True):
    tp, fp, fn, placed, ref = 0, 0, 0, 0, 0
    per_song = []
    for song in songs:
        fin = replay(song, .4, strip)
        ex = song.examples["hazards"]
        queries = candidates(song.base.metadata["duration"], song.base.metadata["bpm"], options)
        pos = (np.asarray(ex.targets["bombs"]) > 0) & (np.asarray(ex.weights["bombs"]) > 0)
        refs = [{"time": queries[i]["time"], "cell": c} for i, c in np.argwhere(pos)]
        preds = [{"time": b["time"], "cell": 4 * b["y"] + b["x"]} for b in fin["bombs"]]
        m = _tolerant_match(refs, preds, .25, ("time",), ("cell",))
        tp += m["true_positive"]; fp += m["false_positive"]; fn += m["false_negative"]
        placed += fin["diagnostics"]["final_bombs"]; ref += len(refs)
        per_song.append(fin["diagnostics"]["final_bombs"])
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn); f1 = 2 * pr * rc / max(1e-9, pr + rc)
    print(f"strip={strip} qthr=.4: per-song={per_song} ref={ref} tp={tp:.0f} fp={fp:.0f} "
          f"P={pr:.3f} R={rc:.3f} F1={f1:.3f}", flush=True)
