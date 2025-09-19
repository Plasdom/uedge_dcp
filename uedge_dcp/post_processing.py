import numpy as np
from uedge import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


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


def compute_lambdaq_extended(Bt, q95, R):
    """G. Federici et al Nucl. Fusion 64 (2024) 036025

    :param Bt: Toroidal field
    :param q95: q95
    :param R: Major radius
    :return: lambda_q
    """

    lambda_q_mm = 0.73 * (Bt**-0.78) * (q95**1.2) * (R**0.1)

    print("\n--- Eich λq Scaling ---")
    print("Formula: λq [mm] = 0.73 x Bt^(-0.78) x q95^(1.2) x R^(0.1)")
    print(
        f"         λq = 0.73 x ({Bt:.3f})^(-0.78) x ({q95:.3f})^(1.2) x ({R:.3f})^(0.1)"
    )
    print(f"         λq ≈ {lambda_q_mm:.2f} mm\n")

    return lambda_q_mm


def compute_lambdaq_mastu_Hmode(Bpol: float, Psol: float) -> float:
    """Compute lambda_q scaling for MAST-U H-mode plasmas (Thornton et al, Plasma Phys. Control. Fusion 56 (2014) 055008 (9pp)))

    :param Bpol: Poloidal magnetic field [T]
    :param Psol: Power crossing the separatrix [MW/m^2???]
    :return lambda_q: Lambda_q [mm]
    """
    lambda_q_mm = 1.84 * Bpol ** (-0.68) * Psol ** (0.18)
    print(r"Predicted λq ≈ {:.2f} mm".format(lambda_q_mm))
    return lambda_q_mm


def compute_lambdaq_bpol(Bpol):
    """
    Compute Eich λq [mm] based on local Bpol in Tesla.
    λq = 0.73 x Bpol^(-1.19)
    """
    lambda_q = 0.63 * Bpol ** (-1.19)
    print("λq = 0.73 x Bpol^(-1.19)")
    print(f"λq ≈ {lambda_q:.2f} mm\n")
    return lambda_q


def q_exp_fit(omp: bool = False):
    """Calculate an exponential fit of the parallel heat flux decay length projected to the outer midplane

    :param omp: Whether to use flux at outer midplane (if False, use flux at outer divertor)
    :return: xq, qparo, qofit, expfun, lqo, omax
    """
    compute_lambdaq_mastu_Hmode(
        com.bpol[bbb.ixmp, com.iysptrx + 1, 3], (bbb.pcoree + bbb.pcorei) / 1e6
    )

    # Bpol_example = 0.5499  # T, from com.bpol[ixmp, iysptrx, 0]
    # compute_lambdaq_bpol(Bpol_example)

    ###R_omp - R_sep (m)
    yyc = com.yyc

    ###R_div - R_sep (m) : Outer divertor
    yyrb = com.yyrb[:, 0]

    q_para_odiv = (bbb.feex[com.ixrb[0], :] + bbb.feix[com.ixrb[0], :]) / com.sxnp[
        com.ixrb[0], :
    ]

    q_para_omp = (bbb.feey[bbb.ixmp, :] + bbb.feiy[bbb.ixmp, :]) / com.sxnp[
        com.ixrb[0], :
    ]

    # q_para_odiv = bbb.ne[com.ixrb[0], :]
    # q_para_omp = bbb.ne[bbb.ixmp, :]
    # q_para_odiv = bbb.te[com.ixrb[0], :] / bbb.ev
    # q_para_omp = bbb.te[bbb.ixmp, :] / bbb.ev

    # q_perp_odiv = (bbb.feey[com.ixrb[0], :] + bbb.feiy[com.ixrb[0], :]) / com.sxnp[
    #     com.ixrb[0], :
    # ]

    s_omp = yyrb

    # select q_para at odiv or omp for the exponential fitting

    if omp is True:
        q_fit = q_para_omp
    else:
        q_fit = q_para_odiv

    interp_fun = interp1d(s_omp, q_fit, kind="cubic", fill_value="extrapolate")
    s_interp = np.linspace(s_omp.min(), s_omp.max(), 300)
    q_interp = interp_fun(s_interp)

    iy = com.iysptrx  # sep
    xq = yyc[iy:-1]
    qparo = q_fit[iy:-1]

    expfun = lambda x, A, lamda_q_inv: A * np.exp(-x * lamda_q_inv)

    try:
        omax = np.argmax(qparo)
        qofit, _ = curve_fit(
            expfun,
            xq[omax:],
            qparo[omax:],
            p0=[np.max(qparo), 1000],
            bounds=(0, np.inf),
        )
        lqo = 1000 / qofit[1]
    except Exception as e:
        print("q_parallel outer fit failed:", e)
        qofit = None
        lqo = 1.0

    return xq, qparo, qofit, expfun, lqo, omax
