"""Assemble the Arm E (feasibility-masked) experimental bundle.

Replaces hazards + arcs in the production v2_expert base with the trained
feasible-20261027-w50-c50 artifacts and their calibration sidecars.
Writes a NEW bundle directory; production v2_expert is untouched.
"""
import sys
from training.pipeline_v2.bundle import assemble_bundle

RUN = "../Data/training/pipeline_v2/runs/feasible-20261027-w50-c50"

document = assemble_bundle(
    "../Data/model/v2_expert",
    replacements={
        "hazards": f"{RUN}/hazards",
        "arcs": f"{RUN}/arcs",
    },
    calibrations={
        "hazards": f"{RUN}/hazards/calibration/calibration.json",
        "arcs": f"{RUN}/arcs/calibration/calibration.json",
    },
    output="../Data/model/v2_expert_feasible-w50-c50",
)
print("assembled:", document.get("model_stages"))
print("data_settings hazard_feasibility_input:",
      document["data_settings"].get("hazard_feasibility_input"))
print("calibrations:", document.get("calibrations"))
sys.exit(0)
