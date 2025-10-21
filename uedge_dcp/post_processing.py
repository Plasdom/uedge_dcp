import numpy as np
from uedge import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erfc


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


def getrrf() -> np.ndarray:
    """Written by Shahinul Islam

    :return: rrf
    """
    bpol_local = 0.5 * (com.bpol[:, :, 2] + com.bpol[:, :, 4])
    bphi_local = 0.5 * (com.bphi[:, :, 2] + com.bphi[:, :, 4])
    btot_local = np.sqrt(bpol_local**2 + bphi_local**2)
    return bpol_local / btot_local


def compute_fx(ixmp: int = None) -> np.ndarray:
    """Written by Shahinul Islam

    :param ixmp: Midplane index, defaults to None
    :return: Flux expansion
    """
    if ixmp is None:
        ixmp = bbb.ixmp

    # Divertor values
    Bpol_div = com.bpol[com.ixrb[0], :, 0]
    Btor_div = com.bphi[com.ixrb[0], :, 0]
    B_div = np.sqrt(Bpol_div**2 + Btor_div**2)

    # Midplane values
    Bpol_omp = com.bpol[ixmp, :, 0]
    Btor_omp = com.bphi[ixmp, :, 0]
    B_omp = np.sqrt(Bpol_omp**2 + Btor_omp**2)

    # Pitch angle at divertor
    sin_theta = Bpol_div / B_div

    # Flux expansion
    fx = (B_div / B_omp) * (1.0 / sin_theta)

    fx_simple = (com.bpol[ixmp, :, 0] * com.rm[ixmp, :, 0]) / (
        com.bpol[com.ixrb[0], :, 0] * com.rm[com.ixrb[0], :, 0]
    )

    return fx, fx_simple


def calculate_flux_expansion(ixmp: int = None):
    """
    Written by Shahinul Islam. Calculate poloidal and tilt flux expansion for the CD configuration.
    D. Moulton et al., Nucl. Fusion 64 (2024) 076049 (31pp)

    Uses:
        com.bpol, com.bphi: 3D arrays of poloidal and toroidal fields
        bbb.ixmp, com.iysptrx, com.nx: indices
        fieldLineAngle(): function returning field angle array

    """
    if ixmp is None:
        ixmp = com.ixmp
    bpol_u = com.bpol[ixmp, com.iysptrx + 1, 0]
    bphi_u = abs(com.bphi[ixmp, com.iysptrx + 1, 0])
    bpol_t = com.bpol[com.nx, com.iysptrx + 1, 0]
    bphi_t = abs(com.bphi[com.nx, com.iysptrx + 1, 0])  # Fixed this line
    # alpha_tilt_deg = fieldLineAngle()
    # alpha_tilt_target = alpha_tilt_deg[com.nx,com.iysptrx+1]
    # alpha_tilt = np.radians(alpha_tilt_target)

    FX_theta = (bpol_u / bphi_u) / (bpol_t / bphi_t)
    # FX_tilt = 1 / np.sin(alpha_tilt)
    return FX_theta


def eich_exp_shahinul_odiv_final(
    omp: bool = False, ixmp: int = None, save_prefix="lambdaq_result"
):
    """Written by Shahinul Islam

    :param omp: Whether to fit at OMP, defaults to False
    :param save_prefix: Save location, defaults to 'lambdaq_result'
    :return: _description_
    """
    yyc = com.yyc.reshape(-1)[:-1]
    if ixmp is None:
        ixmp = bbb.ixmp

    fx = calculate_flux_expansion(ixmp=ixmp)
    fx = np.round(fx)

    # print("fx", fx)

    # === Select which q_parallel to fit (OMP vs ODIV)
    bbb.plateflux()
    ppar = (
        bbb.feex + bbb.feix + 0.5 * bbb.mi[0] * bbb.up[:, :, 0] ** 2 * bbb.fnix[:, :, 0]
    )
    rrf = getrrf()
    q_para_omp = ppar[ixmp, :-1] / com.sx[ixmp, :-1] / rrf[ixmp, :-1]
    q_para_odiv = (
        ppar[com.ixrb[0], :-1] / com.sx[com.ixrb[0], :-1] / rrf[com.ixrb[0], :-1]
    )
    q_data = bbb.sdrrb + bbb.sdtrb
    if "snowflake" in str(com.geometry):
        q_perp_odiv = q_data[:, 0].reshape(-1)[:-1]
    else:
        q_perp_odiv = q_data.reshape(-1)[:-1]

    # s_omp = com.yyrb[:-1]
    q_fit = q_para_omp if omp else q_para_odiv

    # s_omp = s_omp.flatten()
    # q_fit = q_fit.flatten()

    # interp_fun = interp1d(s_omp, q_fit, kind="cubic", fill_value="extrapolate")
    # s_interp = np.linspace(s_omp.min(), s_omp.max(), 300)
    # q_interp = interp_fun(s_interp)
    iy_sep = com.iysptrx + 1

    # === Exponential Fit ===
    xq = yyc[iy_sep:-1]

    qparo = q_fit[iy_sep:-1]
    expfun = lambda x, A, lamda_q_inv: A * np.exp(-x * lamda_q_inv)

    try:
        omax = np.argmax(qparo)
        expfun = lambda x, A, lamda_q_inv: A * np.exp(-x * lamda_q_inv)
        qofit, _ = curve_fit(
            expfun, xq[omax:], qparo[omax:], p0=[np.max(qparo), 100], bounds=(0, np.inf)
        )
        lqo = 1000 / qofit[1]
    except Exception as e:
        print("q_parallel outer fit failed:", e)
        qofit = None
        lqo = 1.0

    xq_omp = xq

    # === Eich Function ===
    def eichFunction(x, S, lq, q0, s0):
        sBar = x - s0
        t0 = (S / (2 * lq * fx)) ** 2
        t1 = sBar / (lq * fx)
        t2 = S / (2 * lq * fx)
        t3 = sBar / S
        q_back = q0 * 1e-3
        return (q0 / 2) * np.exp(t0 - t1) * erfc(t2 - t3) + q_back

    # === Fit Eich Function ===
    if "snowflake" in str(com.geometry):
        yyrb = com.yyrb[:, 0].reshape(-1)[:-1]
    else:
        yyrb = com.yyrb.reshape(-1)[:-1]
    s_omp = yyrb
    q_omp = q_perp_odiv
    interp_fun = interp1d(s_omp, q_omp, kind="cubic", fill_value="extrapolate")
    s_interp = np.linspace(s_omp.min(), s_omp.max(), 300)
    q_interp = interp_fun(s_interp)
    s_fit = s_interp
    q_fit = q_interp
    s0_guess = np.median(s_fit)
    q0_guess = np.max(q_fit)

    p0 = [0.003, 0.002, q0_guess, s0_guess]
    bounds = (
        [0.0005, 0.0005, 1e4, s_fit.min() - 0.01],
        [0.02, 0.02, 1e9, s_fit.max() + 0.01],
    )

    try:
        popt, _ = curve_fit(
            eichFunction, s_fit, q_fit, p0=p0, bounds=bounds, maxfev=10000
        )
        S_fit, lambda_q_fit, q0_fit, s0_fit = popt
        q_fit_full = eichFunction(s_fit, *popt)

        # R
        residuals = q_fit - q_fit_full
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((q_fit - np.mean(q_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
    except Exception as e:
        print("Eich fit failed:", e)
        popt = [0, 0, 0, 0]
        lambda_q_fit = 0.0
        r_squared = 0.0

    return (
        xq,
        qparo,
        qofit,
        expfun,
        omax,
        lqo,
        q_omp,
        s_omp,
        q_fit_full,
        s_fit,
        lambda_q_fit * 1000,
    )


def q_exp_fit_old(omp: bool = False, ixmp=None):
    """Calculate an exponential fit of the parallel heat flux decay length projected to the outer midplane

    :param omp: Whether to use flux at outer midplane (if False, use flux at outer divertor)
    :return: xq, qparo, qofit, expfun, lqo, omax
    """
    if ixmp is None:
        ixmp = bbb.ixmp
        yyc = com.yyc
    else:
        yyc = get_yyc(ixmp=ixmp)
    compute_lambdaq_mastu_Hmode(
        com.bpol[ixmp, com.iysptrx + 1, 3], (bbb.pcoree + bbb.pcorei) / 1e6
    )

    # Bpol_example = 0.5499  # T, from com.bpol[ixmp, iysptrx, 0]
    # compute_lambdaq_bpol(Bpol_example)

    ###R_omp - R_sep (m)
    yyc = yyc[:-1]

    ###R_div - R_sep (m) : Outer divertor
    yyrb = com.yyrb[:, 0]
    rrf = getrrf()

    # q_para_odiv = (bbb.feex[com.ixrb[0], :] + bbb.feix[com.ixrb[0], :]) / com.sxnp[
    #     com.ixrb[0], :
    # ]

    P_par = (
        bbb.feex + bbb.feix + 0.5 * bbb.mi[0] * bbb.up[:, :, 0] ** 2 * bbb.fnix[:, :, 0]
    )
    q_para_odiv = (
        P_par[com.ixrb[0], :-1] / com.sx[com.ixrb[0], :-1] / rrf[com.ixrb[0], :-1]
    )

    # q_para_omp = (bbb.feey[ixmp, :] + bbb.feiy[ixmp, :]) / com.sxnp[ixmp, :]
    # q_para_omp = (bbb.feex[ixmp, :] + bbb.feix[ixmp, :]) / com.sxnp[ixmp, :]
    q_para_omp = q_para_odiv
    # q_para_odiv = bbb.ne[com.ixrb[0], :]
    # q_para_omp = bbb.ne[ixmp, :]
    # q_para_odiv = bbb.te[com.ixrb[0], :] / bbb.ev
    # q_para_omp = bbb.te[ixmp, :] / bbb.ev

    # q_perp_odiv = (bbb.feey[com.ixrb[0], :] + bbb.feiy[com.ixrb[0], :]) / com.sxnp[
    #     com.ixrb[0], :
    # ]

    # s_omp = yyrb
    s_omp = yyc

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
            p0=[np.max(qparo), 1],
            # bounds=[(0, 0), (np.inf, 100)],
            bounds=(0, np.inf),
        )
        lqo = 1000 / qofit[1]
    except Exception as e:
        print("q_parallel outer fit failed:", e)
        qofit = None
        lqo = 1.0

    return xq, qparo, qofit, expfun, lqo, omax


def get_midplane_vals(
    f: np.ndarray, rm: np.ndarray = None, zm: np.ndarray = None, ixmp: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """Find the values of an array at the midplane

    :param f: Variable on UEDGE grid
    :param rm: R grid coords, defaults to None
    :param zm: Z grid coords, defaults to None
    :param ixmp: Midplane index, defaults to None
    :return: x_mp (midplane radial coordinates), f_mp (midplane radial profile)
    """
    if rm is None:
        rm = com.rm
        zm = com.zm
    if ixmp is None:
        ixmp = bbb.ixmp
    x_mp = 0.5 * (rm[ixmp, :, 1] + rm[ixmp, :, 3])
    y_mp = 0.5 * (zm[ixmp, :, 1] + zm[ixmp, :, 3])
    x_1 = rm[ixmp - 1, :, 0]
    y_1 = zm[ixmp - 1, :, 0]
    x_2 = rm[ixmp, :, 0]
    y_2 = zm[ixmp, :, 0]
    d_1 = np.sqrt((x_1 - x_mp) ** 2 + (y_1 - y_mp) ** 2)
    d_2 = np.sqrt((x_2 - x_mp) ** 2 + (y_2 - y_mp) ** 2)
    f_mp = (d_2 * f[ixmp - 1, :] + d_1 * f[ixmp, :]) / (d_1 + d_2)
    return x_mp, f_mp


def get_yyc(
    rm: np.ndarray = None, zm: np.ndarray = None, ixmp: int = None, iysptrx: int = None
) -> np.ndarray:
    """Calculate yyc when UEDGE fails to do so correctly in isudsym=1 cases

    :param rm: R grid coords, defaults to None
    :param zm: Z grid coords, defaults to None
    :param ixmp: Midplane index, defaults to None
    :param iysptrx: Separatrix index, defaults to None
    :return: yyc
    """
    if rm is None:
        rm = com.rm
        zm = com.zm
    if ixmp is None:
        ixmp = bbb.ixmp
    if iysptrx is None:
        iysptrx = com.iysptrx1[0]

    r_mp = rm[ixmp, :, 0]
    z_mp = zm[ixmp, :, 0]

    r_sep = 0.5 * (rm[ixmp, com.iysptrx1[0], 3] + rm[ixmp, com.iysptrx1[0], 4])
    z_sep = 0.5 * (zm[ixmp, com.iysptrx1[0], 3] + zm[ixmp, com.iysptrx1[0], 4])

    y_mp = np.zeros(len(r_mp))
    y_mp[iysptrx + 1] = np.sqrt(
        (r_mp[iysptrx + 1] - r_sep) ** 2 + (z_mp[iysptrx + 1] - z_sep) ** 2
    )
    y_mp[iysptrx] = -np.sqrt(
        (r_mp[iysptrx] - r_sep) ** 2 + (z_mp[iysptrx] - z_sep) ** 2
    )
    for iy in reversed(
        range(
            iysptrx,
        )
    ):
        y_mp[iy] = y_mp[iy + 1] - np.sqrt(
            (r_mp[iy] - r_mp[iy + 1]) ** 2 + (z_mp[iy] - z_mp[iy + 1]) ** 2
        )
    for iy in range(iysptrx + 2, len(y_mp)):
        y_mp[iy] = y_mp[iy - 1] + np.sqrt(
            (r_mp[iy] - r_mp[iy - 1]) ** 2 + (z_mp[iy] - z_mp[iy - 1]) ** 2
        )

    return y_mp
