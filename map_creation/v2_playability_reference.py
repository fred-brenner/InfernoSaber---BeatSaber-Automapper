"""Frozen human cut-axis dwell strata, prepared offline on curated packages.

Source: cut-axis-dwell-app1-random30-20261107/completed-runs.csv.gz.
Highest Standard maps, 51 curated human packages. Sparse 5-7 NPS / 90-degree
strata pool the adjacent 3-5 band. No source corpus is loaded by generation.
"""
SOURCE_SHA256 = 'ba00c26518611a2d40f7a1de3534e57171a5bc60b632f69b5cfa63f9cf515d29'
# key: (map note-span NPS band, hand, ending angle). Values are quartiles,
# fractions, observation counts and source bands (not model permissions).
_REFERENCE = {
    ('<3', 0, 45): {'notes': (1.0, 1.0, 2.0), 'ms': (480.0, 850.446429, 1614.489796), 'one_note': 0.64966, 'five_or_more': 0.091837, 'runs': 294, 'packages': 6, 'source_bands': ('<3',)},
    ('<3', 0, 90): {'notes': (1.0, 2.0, 3.0), 'ms': (697.674419, 1171.875, 1983.471074), 'one_note': 0.481481, 'five_or_more': 0.104377, 'runs': 297, 'packages': 7, 'source_bands': ('<3',)},
    ('<3', 1, 45): {'notes': (1.0, 1.0, 2.0), 'ms': (588.235294, 859.703256, 1696.64919), 'one_note': 0.630137, 'five_or_more': 0.10274, 'runs': 292, 'packages': 7, 'source_bands': ('<3',)},
    ('<3', 1, 90): {'notes': (1.0, 2.0, 3.0), 'ms': (697.674419, 1176.470588, 2040.816327), 'one_note': 0.426621, 'five_or_more': 0.119454, 'runs': 293, 'packages': 7, 'source_bands': ('<3',)},
    ('3-5', 0, 45): {'notes': (1.0, 1.0, 2.0), 'ms': (387.619434, 669.642857, 1250.0), 'one_note': 0.525621, 'five_or_more': 0.098397, 'runs': 3181, 'packages': 27, 'source_bands': ('3-5',)},
    ('3-5', 0, 90): {'notes': (1.0, 1.0, 2.0), 'ms': (483.870968, 774.193548, 1161.290323), 'one_note': 0.538175, 'five_or_more': 0.047486, 'runs': 1074, 'packages': 23, 'source_bands': ('3-5',)},
    ('3-5', 1, 45): {'notes': (1.0, 1.0, 2.0), 'ms': (400.0, 652.173913, 1162.858303), 'one_note': 0.529322, 'five_or_more': 0.096627, 'runs': 3291, 'packages': 27, 'source_bands': ('3-5',)},
    ('3-5', 1, 90): {'notes': (1.0, 1.0, 2.0), 'ms': (468.75, 694.444444, 1171.875), 'one_note': 0.538191, 'five_or_more': 0.050922, 'runs': 1139, 'packages': 24, 'source_bands': ('3-5',)},
    ('5-7', 0, 45): {'notes': (1.0, 2.0, 3.0), 'ms': (342.700286, 585.9375, 1000.0), 'one_note': 0.428329, 'five_or_more': 0.141645, 'runs': 1179, 'packages': 7, 'source_bands': ('5-7',)},
    ('5-7', 0, 90): {'notes': (1.0, 1.0, 2.0), 'ms': (476.190476, 762.711864, 1161.290323), 'one_note': 0.529668, 'five_or_more': 0.056719, 'runs': 1146, 'packages': 27, 'source_bands': ('3-5', '5-7')},
    ('5-7', 1, 45): {'notes': (1.0, 2.0, 3.0), 'ms': (342.846514, 585.9375, 1028.466857), 'one_note': 0.434712, 'five_or_more': 0.141119, 'runs': 1233, 'packages': 7, 'source_bands': ('5-7',)},
    ('5-7', 1, 90): {'notes': (1.0, 1.0, 2.0), 'ms': (468.75, 652.173913, 1171.875), 'one_note': 0.532285, 'five_or_more': 0.054636, 'runs': 1208, 'packages': 28, 'source_bands': ('3-5', '5-7')},
    ('7+', 0, 45): {'notes': (1.0, 2.0, 4.0), 'ms': (230.771872, 461.53494, 882.352941), 'one_note': 0.398756, 'five_or_more': 0.193439, 'runs': 1768, 'packages': 10, 'source_bands': ('7+',)},
    ('7+', 0, 90): {'notes': (1.0, 1.0, 1.0), 'ms': (310.344828, 413.793103, 629.570004), 'one_note': 0.76, 'five_or_more': 0.04, 'runs': 50, 'packages': 5, 'source_bands': ('7+',)},
    ('7+', 1, 45): {'notes': (1.0, 2.0, 3.0), 'ms': (206.896552, 400.0, 799.987793), 'one_note': 0.44461, 'five_or_more': 0.165922, 'runs': 2013, 'packages': 10, 'source_bands': ('7+',)},
    ('7+', 1, 90): {'notes': (1.0, 1.0, 2.0), 'ms': (361.445783, 413.793103, 722.891566), 'one_note': 0.659091, 'five_or_more': 0.068182, 'runs': 44, 'packages': 5, 'source_bands': ('7+',)},
}

def nps_band(nps):
    return '<3' if nps < 3 else '3-5' if nps < 5 else '5-7' if nps < 7 else '7+'


def dwell_reference(nps, hand, angle):
    """Return the supported frozen per-hand, ending-angle distribution."""
    return dict(_REFERENCE[(nps_band(nps), hand, angle)])
