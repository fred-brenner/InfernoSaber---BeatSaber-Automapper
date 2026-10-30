import numpy as np
from training.pipeline_v2 import models
from training.pipeline_v2.settings import training_settings

vocab = {"walls": [{"kind": "none"}, {"kind": "w"}]}
shapes = {"audio": (1, 16), "history": (1, 32, 13), "history_age": (1, 32, 1), "history_mask": (1, 32),
          "conditions": (1, 6), "occupancy": (1, 2, 2, 12), "interval": (1, 2),
          "feasible_wall": (1, 2), "feasible_bombs": (1, 12)}
m = models.build_model("hazards", vocab, input_shapes=shapes, bomb_presence=True)
out = m({k: np.zeros(v, np.float32) for k, v in shapes.items()}, training=False)
print("outputs:", {k: tuple(v.shape) for k, v in out.items()})

targets = {"wall": np.zeros((4,), np.int32), "bombs": np.zeros((4, 12), np.float32)}
weights = {"wall": np.ones((4,), np.float32), "bombs": np.ones((4, 12), np.float32)}
targets["bombs"][0, 3] = targets["bombs"][1, 7] = 1.
pred = m({k: np.zeros((4,) + v[1:], np.float32) for k, v in shapes.items()}, training=False)
comp = models.loss_components("hazards", targets, pred, weights)
print("components:", sorted(comp))
print("loss:", float(models.combine_loss_components("hazards", comp)))

# legacy path still strict
legacy = models.build_model("hazards", vocab, input_shapes=shapes)
pred2 = legacy({k: np.zeros((4,) + v[1:], np.float32) for k, v in shapes.items()}, training=False)
comp2 = models.loss_components("hazards", targets, pred2, weights)
print("legacy components:", sorted(comp2), "loss:", float(models.combine_loss_components("hazards", comp2)))
print("settings flag:", training_settings("hazards").get("bomb_presence"))
