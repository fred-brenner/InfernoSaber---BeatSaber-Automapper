"""Hazard note-space context inputs: emission gating, geometry, model and bundle wiring."""
from types import SimpleNamespace

import numpy as np
import pytest

from training.pipeline_v2.contracts import StageExamples, build_inputs, validate_examples
from training.pipeline_v2.settings import preparation_settings

CONTEXT_KEYS = ("hazard_context_seconds", "hazard_context_back_seconds", "hazard_context_bins")


def _song(duration=60.):
    arrays = {"intensity": np.ones(16), "bin_durations": np.full(16, duration / 16),
              "frame_times": np.arange(64)}
    return SimpleNamespace(metadata={"duration": duration, "bpm": 120.}, arrays=arrays)


def _state():
    return {"notes": [{"id": 0, "time": 1.75, "hand": 0, "x": 1, "y": 1, "d": 3},
                      {"id": 1, "time": 2.6, "hand": 1, "x": 2, "y": 2, "d": 0}],
            "chains": []}


def _options(context=True):
    options = preparation_settings()
    options.pop("hazard_feasibility_input", None)
    if not context:
        for key in CONTEXT_KEYS:
            options["schedule"].pop(key, None)
    return options


def _inputs(options=None):
    candidate = {"time": 2.0, "end_time": 2.5}
    return build_inputs("hazards", _song(), _state(), candidate, [], settings=options or _options())


def test_hazard_context_emitted_with_recorded_spec():
    inputs = _inputs()
    context = inputs["context"]
    assert context.shape == (1, 10, 2, 12)
    # Note at 1.75 s (hand 0, cell 1,1) falls in bin (1.75-2+0.5)/0.25 = 1; note at
    # 2.6 s (hand 1, cell 2,2) falls in bin (2.6-2+0.5)/0.25 = 4.
    assert context[0, 1, 0, 5] == 1.
    assert context[0, 4, 1, 10] == 1.
    assert context.sum() == 2.


def test_hazard_context_omitted_without_spec():
    inputs = _inputs(_options(context=False))
    assert "context" not in inputs
    assert inputs["occupancy"].shape == (1, 2, 2, 12)


def test_hazard_context_shape_validation():
    examples = StageExamples(_inputs(), {"wall": np.zeros(1, np.int32), "bombs": np.zeros((1, 12), np.float32)},
                             {"wall": np.ones(1, np.float32), "bombs": np.ones((1, 12), np.float32)},
                             np.asarray([2.0]))
    validate_examples(examples, "hazards")
    broken = dict(examples.inputs)
    broken["context"] = np.zeros((1, 10, 1, 12), np.float32)
    with pytest.raises(ValueError, match="context"):
        validate_examples(StageExamples(broken, examples.targets, examples.weights, examples.times), "hazards")


def test_hazards_model_signature_follows_input_shapes():
    from training.pipeline_v2.models import build_model

    vocab = {"walls": [{"kind": "none"}, {"kind": "wall"}]}
    legacy = build_model("hazards", vocab)
    assert "context" not in legacy.input_names
    shapes = {key: [1, *value.shape[1:]] for key, value in _inputs().items()}
    model = build_model("hazards", vocab, input_shapes=shapes)
    assert "context" in model.input_names
    assert tuple(model.get_layer("context").output.shape) == (None, 10, 2, 12)


def test_hazard_context_spec_flows_through_stage_data_and_bundles():
    from training.pipeline_v2.bundle import _policy_settings
    from training.pipeline_v2.policies import stage_data

    options = _options()
    data = stage_data(options, "hazards")
    assert [data[key] for key in CONTEXT_KEYS] == [2., .5, 10]
    legacy = stage_data(_options(context=False), "hazards")
    assert all(key not in legacy for key in CONTEXT_KEYS)

    settings = {}
    identity = {"stage": "hazards", "stage_data": data}
    _policy_settings(settings, identity)
    assert settings["schedule"]["hazard_context_seconds"] == 2.
    assert settings["schedule"]["hazard_context_bins"] == 10


def _feasibility_options():
    options = _options()
    options["hazard_feasibility_input"] = True
    return options


def test_hazard_feasibility_inputs_emitted_and_gated():
    candidate = {"time": 2., "end_time": 2.5, "kind": "wall",
                 "_feasible_wall": np.asarray([True, True, False]),
                 "_feasible_bombs": np.r_[np.ones(6, bool), np.zeros(6, bool)]}
    inputs = build_inputs("hazards", _song(), _state(), candidate, [], settings=_feasibility_options())
    assert inputs["feasible_wall"].tolist() == [[1., 1., 0.]]
    assert inputs["feasible_bombs"].shape == (1, 12)
    assert inputs["feasible_bombs"].sum() == 6.
    inputs = _inputs()
    assert "feasible_wall" not in inputs and "feasible_bombs" not in inputs


def test_hazard_feasibility_inputs_require_masks():
    with pytest.raises(ValueError, match="masks"):
        build_inputs("hazards", _song(), _state(), {"time": 2., "end_time": 2.5}, [],
                     settings=_feasibility_options())


def test_hazard_feasibility_heads_mask_impossible_options():
    from training.pipeline_v2.models import build_model

    vocab = {"walls": [{"kind": "none"}, {"kind": "wall"}, {"kind": "wall"}]}
    candidate = {"time": 2., "end_time": 2.5, "kind": "wall",
                 "_feasible_wall": np.asarray([True, False, True]),
                 "_feasible_bombs": np.zeros(12, bool)}
    inputs = build_inputs("hazards", _song(), _state(), candidate, [], settings=_feasibility_options())
    shapes = {key: [1, *value.shape[1:]] for key, value in inputs.items()}
    model = build_model("hazards", vocab, input_shapes=shapes)
    assert {"feasible_wall", "feasible_bombs"} <= set(model.input_names)
    outputs = model({key: value for key, value in inputs.items()})
    wall, bombs = outputs["wall"].numpy()[0], outputs["bombs"].numpy()[0]
    assert wall[1] < 1e-3 and np.isclose(wall.sum(), 1.)
    assert not bombs.any()
    legacy = build_model("hazards", vocab)
    assert "feasible_wall" not in legacy.input_names


def test_hazard_feasibility_flag_flows_through_stage_data_and_bundles():
    from training.pipeline_v2.bundle import _policy_settings
    from training.pipeline_v2.policies import stage_data

    data = stage_data(_feasibility_options(), "hazards")
    assert data["hazard_feasibility_input"] is True
    assert "hazard_feasibility_input" not in stage_data(_options(), "hazards")
    settings = {}
    _policy_settings(settings, {"stage": "hazards", "stage_data": data})
    assert settings["hazard_feasibility_input"] is True
