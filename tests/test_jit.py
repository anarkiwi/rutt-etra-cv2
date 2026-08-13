"""Check the compiled kernel against the interpreted one.

The rest of the suite runs with NUMBA_DISABLE_JIT so coverage can see inside
the kernels; this runs the real compiled path in a subprocess and compares.
"""

import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent("""
    import numpy as np
    from ruttetra.core import ScanParams
    from ruttetra.raster import render

    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    for beam in ("rate", "speed"):
        image = render(frame, ScanParams(lines=6, gain=0.3, beam=beam))
        print(beam, image.shape, int(image.sum()), int(image.max()))
    """)


def _run(disable_jit):
    """Run the reference script with JIT on or off."""
    env = dict(os.environ, PYTHONPATH=os.getcwd(), NUMBA_DISABLE_JIT=disable_jit)
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return result.stdout


def test_compiled_kernel_matches_interpreted():
    """Numba compilation must not change the picture."""
    assert _run("1") == _run("0")
