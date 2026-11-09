"""Combined-geometry scorer; geometry_version identifies the current definition."""

import numpy as np

from evaluation import playability_metrics as legacy
from map_creation.v2_playability_costs import (
    VERSION, interference_severity, occlusion_severity,
)


def geometry_events(prepared):
    """Return zero-inclusive M2 note and M4 eligible-pair severities."""
    m2, m4 = [], []
    for index, note in enumerate(prepared):
        before, dt = legacy._nearest_occluder(prepared, index)
        m2.append(occlusion_severity(note, before, dt))
    volumes = [legacy.swing_volume(note) for note in prepared]
    for index, left in enumerate(prepared):
        if left["hand"] != 0:
            continue
        for partner in legacy._cross_hand_partners(prepared, index):
            right = prepared[partner]
            m4.append(interference_severity(left, right, abs(left["t"] - right["t"]),
                                            volumes[index], volumes[partner]))
    return {"m2": np.asarray(m2, dtype=float), "m4": np.asarray(m4, dtype=float)}


def score_map(notes):
    """Score all metrics, replacing only M2/M4 with the combined definitions.

    ``events`` include zeros for the approved human trigger denominators.
    The original positive-only ``units`` and scalar output contracts remain.
    """
    notes = list(notes)
    prepared = legacy.prepare_notes(notes)
    scores = legacy.score_map(notes)
    events = geometry_events(prepared)
    for metric, values in events.items():
        scores[metric] = {"value": float(values.mean()) if len(values) else 0.,
                          "units": values[values > 0], "events": values,
                          "incidents": []}
    minutes = scores["span_seconds"] / 60
    scores["m4"]["value"] = float(events["m4"].sum() / minutes) if minutes else 0.
    m2_incidents, m4_incidents = [], []
    for index, note in enumerate(prepared):
        before, dt = legacy._nearest_occluder(prepared, index)
        value = events["m2"][index]
        if value >= .5 - 1e-9:
            m2_incidents.append({"t": note["t"], "note": (note["x"], note["y"]),
                                 "occluder": (before["x"], before["y"]), "dt": dt, "occl": float(value)})
    volumes = [legacy.swing_volume(note) for note in prepared]
    offset = 0
    for index, left in enumerate(prepared):
        if left["hand"] != 0:
            continue
        for partner in legacy._cross_hand_partners(prepared, index):
            right = prepared[partner]
            severity = float(events["m4"][offset])
            offset += 1
            if severity <= 0:
                continue
            dt = abs(left["t"] - right["t"])
            m4_incidents.append({"t": max(left["t"], right["t"]), "dt": dt,
                                 "a": (left["x"], left["y"], left["d"]),
                                 "b": (right["x"], right["y"], right["d"]),
                                 "cells": sorted(volumes[index] & volumes[partner]),
                                 "severity": severity,
                                 "crossover": max(0., left["x"] - right["x"]) * max(0., 1 - dt / legacy.TAU_X)})
    scores["m2"]["incidents"] = sorted(m2_incidents, key=lambda event: event["occl"], reverse=True)[:legacy.INCIDENT_TOP_K]
    scores["m4"]["incidents"] = sorted(m4_incidents, key=lambda event: event["severity"], reverse=True)[:legacy.INCIDENT_TOP_K]
    scores["geometry_version"] = VERSION
    return scores
