import numpy as np
from numba import njit
from tqdm import tqdm

from phasenettfps.beamforming.core.bf import FreqBF


def brute_force_search(
    bf: FreqBF,
    phis,
    thetas,
    vs,
    desc,
):
    # Initialize max value and corresponding parameters
    max_value = -np.inf
    max_iphi = max_itheta = max_iv = np.nan

    # Iterate through all combinations of phi, theta, and v
    res = np.zeros((len(phis), len(thetas), len(vs)), dtype=np.float64)
    for iphi, phi in tqdm(enumerate(phis), total=len(phis), desc=desc):
        for itheta, theta in enumerate(thetas):
            for iv, v in enumerate(vs):
                current_value = bf.opt_func((phi, theta, v))
                res[iphi, itheta, iv] = current_value
                if current_value > max_value:
                    max_value = current_value
                    max_iphi = iphi
                    max_itheta = itheta
                    max_iv = iv

    return max_iphi, max_itheta, max_iv, max_value, res
