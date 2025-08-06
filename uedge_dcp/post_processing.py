import numpy as np
from uedge import *


def mask_guard_cells(variable: np.ndarray):
    """Set all guard cell values to nans

    :param variable: Input variable
    :return: New array with guard cells set to nan
    """
    v = variable.copy()
    v[0, :] = np.nan
    v[-1, :] = np.nan
    v[:, 0] = np.nan
    v[:, -1] = np.nan
    if com.nxpt == 2:
        v[com.ixrb[0] + 1, :] = np.nan
        v[com.ixlb[1], :] = np.nan

    return v


def getomit(var):
    """Helper function to handled partial grids w/ omits"""
    nxomit = 0
    nyomit = 0
    if isinstance(var, str):
        var = var
    if nyomit > 0:
        var = var[:, :-nyomit]
    return var[nxomit:]


def getThetaHat(ix, iy):
    """Get the unit vector pointing on the poloidal direction in R-Z coordinates

    :param ix: x index
    :param iy: y index
    :return: theta_hat
    """
    p1R = com.rm[ix, iy, 1]
    p1Z = com.zm[ix, iy, 1]
    p2R = com.rm[ix, iy, 2]
    p2Z = com.zm[ix, iy, 2]
    dR = p2R - p1R
    dZ = p2Z - p1Z
    mag = (dR**2 + dZ**2) ** 0.5
    theta_hat = np.array([dR / mag, dZ / mag])
    return theta_hat


def getrHat(ix, iy):
    """Get the unit vector pointing in the radial direction in R-Z coordinates

    :param ix: x index
    :param iy: y index
    :return: r_hat
    """
    dR = com.rm[ix, iy, 2] - com.rm[ix, iy, 1]
    dZ = com.zm[ix, iy, 2] - com.zm[ix, iy, 1]
    mag = (dR**2 + dZ**2) ** 0.5
    r_hat = np.array([-dZ / mag, dR / mag])
    return r_hat


def getVectorRZ(vx, vy):
    """Convert an input vector in UEDGE (x-y) coordinates into R-Z coordinates

    :param vx: x-component of a vector
    :param vy: y-component of a vector
    :return: vR, vZ
    """
    vec = np.zeros((com.nx + 2, com.ny + 2, 2))
    for ix in range(1, com.nx + 1):
        for iy in range(1, com.ny + 1):
            vec[ix, iy] = vx[ix, iy] * np.array(getThetaHat(ix, iy))
            vec[ix, iy] += vy[ix, iy] * np.array(getrHat(ix, iy))
    vR = vec[:, :, 0]
    vZ = vec[:, :, 1]

    return vR, vZ


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
    """Get the ExB and grad B convective heat fluxes. Outputs have dimensions [com.nx+2,com.ny+2,2], where the third dimension contains the x and y components of the vector (in UEDGE coordinates)

    :return: q_ExB, q_gradB
    """
    # Compute the heat fluxes
    q_ExB = np.zeros((com.nx + 2, com.ny + 2, 2))
    q_gradB = np.zeros((com.nx + 2, com.ny + 2, 2))
    p = (bbb.ne * bbb.te) + (bbb.ni[:, :, 0] * bbb.ti)
    q_ExB[:, :, 0] = -np.sign(bbb.b0) * np.sqrt(1 - com.rr**2) * bbb.v2ce[:, :, 0] * p
    q_ExB[:, :, 1] = bbb.vyce[:, :, 0] * p
    q_gradB[:, :, 0] = -np.sign(bbb.b0) * np.sqrt(1 - com.rr**2) * bbb.v2cb[:, :, 0] * p
    q_gradB[:, :, 1] = bbb.vycb[:, :, 0] * p

    return q_ExB, q_gradB
