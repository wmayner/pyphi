import numpy as np
import pyphi


def map_to_floor_and_ceil(y, floor, ceiling):
    return floor + (ceiling - floor) * y


def reshape_to_md(tpm):
    N = int(np.log2(len(tpm)))
    try:
        return tpm.reshape([2] * N + [1], order="F").astype(float)
    except:
        return pyphi.convert.to_md(tpm)
