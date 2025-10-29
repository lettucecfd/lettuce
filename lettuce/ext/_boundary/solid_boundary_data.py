
import numpy as np

__all__ = ['SolidBoundaryData']

# index lists for Bounce Back Boundaries: populations for FWBB, HWBB; IBB.
#   - d and the differentiation of lt (less than 0.5) and gt (greater than 0.5) is only for IBB
class SolidBoundaryData(dict):
    f_index_lt: np.ndarray
    f_index_gt: np.ndarray
    d_lt: np.ndarray
    d_gt: np.ndarray
    points_inside: np.ndarray
    solid_mask: np.ndarray
    not_intersected: np.ndarray = np.ndarray([])