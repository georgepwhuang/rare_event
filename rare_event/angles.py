import numpy as np
from pyqsp import angle_sequence
from pyqsp.poly import PolyTaylorSeries
from scipy.special import erf

def convert_qsvt_angles(angles):
    num_angles = len(angles)
    update_vals = np.zeros(num_angles)

    update_vals[0] = - np.pi / 4 * (2 * num_angles - 1)
    update_vals[1:-1] = np.pi / 2
    update_vals[-1] = np.pi / 4

    return angles + update_vals

def generate_thresh_angles(deg, delta, threshold):

    #Generate projector angles
    func = lambda x: (erf(delta * (x + threshold)) - erf(delta * (x - threshold))) / 2

    poly = PolyTaylorSeries().taylor_series(
        func=func,
        degree=deg,
        max_scale=0.99,
        chebyshev_basis=True,
        cheb_samples=2*deg)

    pcoefs = poly.coef
    # force odd coefficients to be zero, since the polynomial must be even
    pcoefs[1::2] = 0

    (proj_set, _, _) = angle_sequence.QuantumSignalProcessingPhases(
        pcoefs,
        method='sym_qsp',
        chebyshev_basis=True,
        signal_operator="Wx")

    proj_set = convert_qsvt_angles(proj_set)
    
    return proj_set