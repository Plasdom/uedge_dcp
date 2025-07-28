import numpy as np
from uedge import *


def get_dr_plate(r):
    """Get the radial grid spacings for a given set of radial locations

    :param r: Radial coordinates
    :return: dr
    """
    dr = np.zeros(len(r))
    for i in range(1, len(r) - 1):
        dr[i] = 0.5 * (r[i + 1] - r[i]) + 0.5 * (r[i] - r[i - 1])
    dr[0] = 2 * (r[1] - r[0])
    dr[-1] = 2 * (r[-1] - r[-2])
    return dr


def get_Q_target_proportions():
    """Get the proportions of heat flux delivered to each strike point

    :return: P1, P2, P3, P4
    """
    bbb.plateflux()
    q_odata = (bbb.sdrrb + bbb.sdtrb).T
    q_idata = (bbb.sdrlb + bbb.sdtlb).T
    q1 = q_odata[0]
    q2 = q_odata[1]
    q3 = q_idata[1]
    q4 = q_idata[0]
    r1 = com.yyrb.T[0]
    r2 = com.yyrb.T[1]
    r3 = com.yylb.T[1]
    r4 = com.yylb.T[0]
    dr1 = get_dr_plate(r1)
    dr2 = get_dr_plate(r2)
    dr3 = get_dr_plate(r3)
    dr4 = get_dr_plate(r4)

    P1 = np.sum(q1[1:-1] * dr1[1:-1])
    P2 = np.sum(q2[1:-1] * dr2[1:-1])
    P3 = np.sum(q3[1:-1] * dr3[1:-1])
    P4 = np.sum(q4[1:-1] * dr4[1:-1])

    return P1, P2, P3, P4


def get_q_drifts():
    """Get the ExB and grad B convective heat fluxes

    :return: q_ExB, q_gradB
    """
    # Compute the heat fluxes
    q_ExB = bbb.vyce[:, :, 0] * ((bbb.ne * bbb.te) + (bbb.ni[:, :, 0] * bbb.ti))
    q_gradB = bbb.vycb[:, :, 0] * ((bbb.ne * bbb.te) + (bbb.ni[:, :, 0] * bbb.ti))

    # Set to zero on boundaries
    q_ExB[com.ixlb[0], :] = np.nan
    q_ExB[com.ixlb[1], :] = np.nan
    q_ExB[com.ixrb[0] + 1, :] = np.nan
    q_ExB[com.ixrb[1] + 1, :] = np.nan

    q_gradB[com.ixlb[0], :] = np.nan
    q_gradB[com.ixlb[1], :] = np.nan
    q_gradB[com.ixrb[0] + 1, :] = np.nan
    q_gradB[com.ixrb[1] + 1, :] = np.nan

    return q_ExB, q_gradB
