"""Shared fixtures.

JIT is disabled so coverage can see inside the numba kernels; test_jit.py
re-runs the compiled path in a subprocess and checks the two agree.
"""

import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# pylint: disable=wrong-import-position
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

# One OpenCV thread per worker; its default pool oversubscribes badly under xdist.
cv2.setNumThreads(1)


@pytest.fixture(name="step")
def step_fixture():
    """Frame split into a dark and a bright half."""
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, 32:] = 240
    frame[:, :32] = 16
    return frame


@pytest.fixture(name="noise")
def noise_fixture():
    """Deterministic random frame."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)


@pytest.fixture(name="ramp")
def ramp_fixture():
    """Horizontal luminance ramp from black to white."""
    row = np.linspace(0, 255, 64, dtype=np.uint8)
    return np.repeat(np.tile(row, (48, 1))[:, :, None], 3, axis=2)
