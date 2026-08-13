"""Laser primary colorimetry.

CIE 1931 2 degree matching functions via the multi-lobe Gaussian fit of Wyman,
Sloan & Shirley, JCGT 2(2), 2013, which is within about 1% of the tabulated
data and avoids shipping a table.
"""

import numpy as np

D65 = (0.31272, 0.32903)
LUMENS_PER_WATT = 683.0

# sRGB primaries, IEC 61966-2-1.
XYZ_TO_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)


def _lobe(wavelength, peak, low, high):
    """Piecewise Gaussian with a different width either side of the peak."""
    spread = np.where(wavelength < peak, low, high)
    return np.exp(-0.5 * ((wavelength - peak) / spread) ** 2)


def matching(wavelength):
    """CIE 1931 2 degree x, y, z matching functions at a wavelength in nm."""
    nanometres = np.asarray(wavelength, dtype=np.float64)
    x = (
        1.056 * _lobe(nanometres, 599.8, 37.9, 31.0)
        + 0.362 * _lobe(nanometres, 442.0, 16.0, 26.7)
        - 0.065 * _lobe(nanometres, 501.1, 20.4, 26.2)
    )
    y = 0.821 * _lobe(nanometres, 568.8, 46.9, 40.5) + 0.286 * _lobe(
        nanometres, 530.9, 16.3, 31.1
    )
    z = 1.217 * _lobe(nanometres, 437.0, 11.8, 36.0) + 0.681 * _lobe(
        nanometres, 459.0, 26.0, 13.8
    )
    return np.stack([x, y, z], axis=-1)


def efficacy(wavelength):
    """Luminous efficacy of a monochromatic source, lumens per optical watt."""
    return LUMENS_PER_WATT * matching(wavelength)[..., 1]


def primaries(wavelengths):
    """XYZ per optical watt for each laser primary, as a 3x3 matrix of columns."""
    return matching(np.asarray(wavelengths, dtype=np.float64)).T


def white_balance(wavelengths, white=D65):
    """Optical watt ratios that put the three primaries on a white point.

    Returned normalised so the largest ratio is 1, which is the channel that
    limits the achievable white.
    """
    target = np.array(
        [white[0] / white[1], 1.0, (1.0 - white[0] - white[1]) / white[1]]
    )
    watts = np.linalg.solve(primaries(wavelengths), target)
    if np.any(watts <= 0):
        raise ValueError("these primaries cannot reach that white point")
    return watts / watts.max()


def drive_limits(wavelengths, power, white=D65):
    """Per channel drive in [0, 1] that white balances a projector.

    power is the available optical milliwatts per channel. The channel needing
    the largest share relative to what it has is pinned at full drive.
    """
    watts = np.asarray(power, dtype=np.float64)
    if np.any(watts <= 0):
        raise ValueError("every channel needs some power")
    need = white_balance(wavelengths, white)
    drive = need / watts
    return drive / drive.max()


def to_display(power, wavelengths, exposure=1.0):
    """Convert per channel optical watts to display sRGB.

    Shows what the beam would look like, so an uncalibrated projector renders
    the blue violet its power split actually produces.
    """
    watts = np.asarray(power, dtype=np.float64)
    xyz = watts @ primaries(wavelengths).T
    linear = xyz @ XYZ_TO_SRGB.T
    linear = np.clip(linear * exposure, 0.0, None)
    peak = linear.max()
    if peak > 1.0:
        linear = linear / peak
    return np.clip(1.055 * np.power(linear, 1 / 2.4) - 0.055, 0.0, 1.0)


def chromaticity(power, wavelengths):
    """CIE 1931 x, y of a mix of the primaries."""
    xyz = np.asarray(power, dtype=np.float64) @ primaries(wavelengths).T
    total = xyz.sum(axis=-1, keepdims=True)
    return (xyz / np.where(total == 0, 1.0, total))[..., :2]
