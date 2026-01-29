from tkinter import NO
from xml.etree.ElementInclude import include
import numpy as np
from uedge import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erfc
from uedge_dcp.gridue_manip import UESave
import uetools


def get_power_flows(c: uetools.Case | None = None):
    """Calculate the energy flux channels leaving the simulation domain (via the outer wall, the target apltes, the PFR(s), and via volume radiation)

    :param c: Uetools case
    :return: Pin, Prad, Pplates, Pwall, Ppfr
    """
    # pbindy = np.sum(c.get("fniy") * 13.6 * 1.6022e-19, axis=2)
    # mi = c.get("minu") * c.get("mp")
    # f = c.get("feey") + c.get("feiy") + pbindy
    # if c.get("isphion") == 1:
    #     f += c.get("fqy") * c.get("phi")
    # for isp in range(c.get("ni").shape[-1]):
    #     f += np.sum(0.5 * mi[isp] * c.get("up", isp) ** 2 * c.get("fniy", isp))

    # Pin = c.get("pcoree") + c.get("pcorei")

    # try:
    #     Prad = np.sum((c.get("prad") + c.get("pradhyd")) * c.get("vol"))
    # except:
    #     Prad = np.sum(c.get("pradhyd") * c.get("vol"))

    # P1, P2, P3, P4 = get_Q_plates(c=c, include_radiation=False, verbose=False)
    # Pplates = P1 + P2 + P3 + P4

    # Pwall = np.sum(f[: c.get("ixrb")[0] + 1, c.get("ny")])

    # pfr_flux = []
    # isixcore = get_isixcore(c)
    # for ix in range(c.get("ixrb")[0] + 1, c.get("nx")):
    #     pfr_flux.append(f[ix, c.get("ny")])
    # for ix in range(c.get("nx")):
    #     if isixcore[ix] == 0:
    #         pfr_flux.append(-f[ix, 0])
    # Ppfr = np.sum(pfr_flux)

    # print(" P_in [MW]    | P_plates [MW]| P_rad [MW]   | P_wall [MW]  | P_pfr [MW]  ")
    # print(
    #     " {:.2e}     | {:.2e}     | {:.2e}     | {:.2e}     | {:.2e}".format(
    #         Pin / 1e6, Pplates / 1e6, Prad / 1e6, Pwall / 1e6, Ppfr / 1e6
    #     )
    # )
    # print(
    #     "    Delta = {:+.2f} MW (= {:+.2f}% of P_in)".format(
    #         -(Pin - Pplates - Prad - Pwall - Ppfr) / 1e6,
    #         -100 * (Pin - Pplates - Prad - Pwall - Ppfr) / Pin,
    #     )
    # )

    # return Pin, Prad, Pplates, Pwall, Ppfr

    if c is None:
        feiy = bbb.feiy 
        feey = bbb.feey 
        ny = com.ny 
        sy = com.sy 
        minu = bbb.minu 
        mp = bbb.mp 
        up = bbb.up 
        fniy = bbb.fniy
        ni = bbb.ni
        fqy = bbb.fqy
        phi = bbb.phi 
        prad = bbb.prad 
        pradhyd = bbb.pradhyd
        vol = com.vol
        pcoree = bbb.pcoree
        pcorei = bbb.pcorei
    else:
        feiy = c.get("feiy")
        feey = c.get("feey")
        ny = c.get("ny")
        sy = c.get("sy")
        minu = c.get("minu") 
        mp = c.get("mp") 
        up = c.get("up") 
        fniy = c.get("fniy")
        ni = c.get("ni")
        fqy = c.get("fqy")
        phi = c.get("phi") 
        prad = c.get("prad") 
        pradhyd = c.get("pradhyd")
        vol = c.get("vol")
        pcoree = c.get("pcoree")
        pcorei = c.get("pcorei")


    ebind = 13.6
    ev = 1.6022e-19
    fwall_nx = (feiy + feey)[:,ny] / sy[:,ny]
    # return (feiy + feey)[:,ny]
    for isp in range(ni.shape[-1]):
        fwall_nx += (
            0.5
            * minu[isp]
            * mp
            * up[:,ny,isp] ** 2
            * fniy[:,ny,isp]
            / sy[:, ny]
        )
    fwall_nx += (fniy[:,ny,0] * ebind * ev) / sy[
        :, ny
    ]
    try:
        fwall_nx += (
            fqy[:, ny] * phi[:, ny + 1]
        ) / sy[:, ny]
    except:
        print("No fqy found.")
    fwall_nx[0] = fwall_nx[1]
    fwall_nx[-1] = fwall_nx[-2]

    fwall_0 = (feiy + feey)[:, 0] /sy[:, 0]
    for isp in range(ni.shape[-1]):
        fwall_0 += (
            0.5
            * minu[isp]
            * mp
            * up[:, 0, isp] ** 2
            * fniy[:, 0, isp]
            / sy[:, 0]
        )
    fwall_0 += (fniy[:, 0,0] * ebind * ev) / sy[:, 0]
    try:
        fwall_0 += (fqy[:, 0] * phi[:, 1]) / sy[:, 0]
    except:
        print("No fqy found.")
    fwall_0[0] = fwall_0[1]
    fwall_0[-1] = fwall_0[-2]

    Pin = pcoree + pcorei
    # print(fwall_nx)
    # print(fwall_0)
    # print(np.sum(fwall_nx * sy[:, ny]), np.sum(fwall_0 * sy[:, 0]) - Pin)
    Pwall = np.sum(fwall_nx * sy[:, ny])
    Pwall += np.sum(fwall_0 * sy[:, 0]) - Pin
    try:
        Prad = np.sum((prad + pradhyd) * vol)
    except:
        print("No prad found.")
        Prad = np.sum(pradhyd * vol)
    
    P_SPs = get_Q_plates(c=c, include_radiation=False, verbose=False)
    Pplates = sum(P_SPs)

    print(" P_in [MW]    | P_plates [MW]| P_rad [MW]   | P_wall [MW]  ")
    print(
        " {:.2e}     | {:.2e}     | {:.2e}     | {:.2e}     ".format(
            Pin / 1e6, Pplates / 1e6, Prad / 1e6, Pwall / 1e6
        )
    )
    print(
        "    Delta = {:+.2f} MW (= {:+.2f}% of P_in)".format(
            -(Pin - Pplates - Prad - Pwall) / 1e6,
            -100 * (Pin - Pplates - Prad - Pwall) / Pin,
        )
    )
    return Pin, Prad, Pplates, Pwall


def get_isixcore(c: uetools.Case):
    """Get isixcore array. Needed when this variable is not saved in hdf5 output.

    :param c: Case
    :return: array(nx+2)
    """
    geometry = c.get("geometry")[0].decode("UTF-8")
    ixpt1 = c.get("ixpt1")
    ixpt2 = c.get("ixpt2")
    nx = c.get("nx")
    isixcore = np.zeros(nx + 2, dtype=int)

    if "snowflake15" in geometry:
        for ix in range(nx + 2):
            if (ix > ixpt1[0] and ix <= ixpt2[0]) or (ix > ixpt1[1] and ix <= ixpt2[1]):
                isixcore[ix] = 1
    elif "snowflake45" in geometry:
        for ix in range(nx + 2):
            if ix > ixpt1[0] and ix <= ixpt2[0]:
                isixcore[ix] = 1
    elif "snowflake75" in geometry:
        for ix in range(nx + 2):
            if ix > ixpt1[0] and ix <= ixpt2[0]:
                isixcore[ix] = 1
    elif "snowflake105" in geometry:
        for ix in range(nx + 2):
            if ix > ixpt2[0] and ix <= ixpt1[1]:
                isixcore[ix] = 1
    elif "snowflake135" in geometry:
        for ix in range(nx + 2):
            if ix > ixpt2[0] and ix <= ixpt1[1]:
                isixcore[ix] = 1
    elif "snowflake165" in geometry:
        for ix in range(nx + 2):
            if (ix > ixpt1[0] and ix <= ixpt2[0]) or (ix > ixpt1[1] and ix <= ixpt2[1]):
                isixcore[ix] = 1

    return isixcore


def get_flux_tube(
    var: np.ndarray,
    iy: int,
    ix_seed: int = 0,
    direction: str = "p",
    xaxis: str = "pol",
    c: uetools.Case | None = None,
):
    """Get variables and coordinates along a single flux tube on UEDGE grid

    :param var: _description_
    :param iy: _description_
    :param ix_seed: _description_, defaults to 0
    :param direction: _description_, defaults to "p"
    :param xaxis: _description_, defaults to "pol"
    :param c: _description_, defaults to None
    :return: _description_
    """

    # Retrieve grid variables
    if c is None:
        nx = com.nx
        ixp1 = bbb.ixp1
        ixm1 = bbb.ixm1
        dx = com.dx
        rr = com.rr
    else:
        nx = c.get("nx")
        ixp1 = c.get("ixp1")
        ixm1 = c.get("ixm1")
        dx = 1 / c.get("gx")
        bpol = c.get("bpol")
        b = c.get("b")
        rr = 0.25 * (
            bpol[:, :, 1] / b[:, :, 1]
            + bpol[:, :, 2] / b[:, :, 2]
            + bpol[:, :, 3] / b[:, :, 3]
            + bpol[:, :, 4] / b[:, :, 4]
        )

    # Get the indices of the flux tube
    posx = ix_seed
    posx_arr = []
    for _ in range(1, nx + 2):
        posx_prev = posx
        if direction == "p":
            posx = ixp1[posx_prev, iy]
        else:
            posx = ixm1[posx_prev, iy]
        # Identify end of branch
        if posx == posx_prev:
            break
        posx_arr.append(posx)
    posx_arr = np.array(posx_arr)

    # Retrieve variables along the flux tube
    plotvar = np.zeros(len(posx_arr))
    xpar = np.zeros(len(posx_arr))
    xpol = np.zeros(len(posx_arr))
    xind = []
    plotvar[0] = var[0, iy]
    xpar[0] = dx[0, iy] / rr[0, iy]
    xpol[0] = dx[0, iy]
    xind.append("0")
    for ix, posx in enumerate(posx_arr):
        plotvar[ix] = var[posx, iy]
        xpar[ix] = xpar[ix - 1] + dx[posx, iy] / rr[posx, iy]
        xpol[ix] = xpol[ix - 1] + dx[posx, iy]
        xind.append(str(posx))

    if xaxis == "pol":
        x = xpol
    elif xaxis == "par":
        x = xpar
    elif xaxis == "index":
        x = xind

    return posx_arr, x, plotvar


def restore_flipped(savepath, vars=["te", "phi", "ti", "ni", "ng", "up"]):
    """For snowflake cases, restore a UEDGE save file from a mirrored configuration. For example, the mirror configuration to a SF15 is a SF165.

    Mirror pairs:
        - SF15 / SF165
        - SF45 / SF135
        - SF75 / SF105

    :param savepath: Filepath of UEDGE hdf5 save file
    """

    save = UESave(savepath)
    if "te" in vars:
        bbb.tes[: com.ixrb[0] + 2, :] = np.flip(
            save.vars["te"][: com.ixrb[0] + 2, :], axis=0
        )
        bbb.tes[com.ixlb[1] :, :] = np.flip(save.vars["te"][com.ixlb[1] :, :], axis=0)
    if "phi" in vars:
        bbb.phis[: com.ixrb[0] + 2, :] = np.flip(
            save.vars["phi"][: com.ixrb[0] + 2, :], axis=0
        )
        bbb.phis[com.ixlb[1] :, :] = np.flip(save.vars["phi"][com.ixlb[1] :, :], axis=0)
    if "ti" in vars:
        bbb.tis[: com.ixrb[0] + 2, :] = np.flip(
            save.vars["ti"][: com.ixrb[0] + 2, :], axis=0
        )
        bbb.tis[com.ixlb[1] :, :] = np.flip(save.vars["ti"][com.ixlb[1] :, :], axis=0)
    if "ni" in vars:
        bbb.nis[: com.ixrb[0] + 2, :, :] = np.flip(
            save.vars["ni"][: com.ixrb[0] + 2, :, :], axis=0
        )
        bbb.nis[com.ixlb[1] :, :, :] = np.flip(
            save.vars["ni"][com.ixlb[1] :, :, :], axis=0
        )
    if "ng" in vars:
        bbb.ngs[: com.ixrb[0] + 2, :] = np.flip(
            save.vars["ng"][: com.ixrb[0] + 2, :], axis=0
        )
        bbb.ngs[com.ixlb[1] :, :] = np.flip(save.vars["ng"][com.ixlb[1] :, :], axis=0)
    if "up" in vars:
        bbb.ups[: com.ixrb[0] + 2, :] = np.flip(
            save.vars["up"][: com.ixrb[0] + 2, :], axis=0
        )
        bbb.ups[com.ixlb[1] :, :] = np.flip(save.vars["up"][com.ixlb[1] :, :], axis=0)


def get_pfr_vals_sf45(f):
    """Get the values of input array f at the PFR boundaries. f is assumed to be a flux variable defined on the north face of cells

    :param f:  a flux variable defined on the north face of cells
    :return: Array of values of f on the PFR boundaries
    """
    pfr1 = f[com.ixrb[0] + 1 :, com.ny, ...]
    pfr23 = f[np.where(com.isixcore == 0), 0, ...][0, ...]
    pfr_vals = np.concatenate([pfr1, pfr23])
    return pfr_vals


def get_xpt_positions():
    """Get positions of X-point(s)

    :return: R_xpt[xpt1, ...], Z_xpt[xpt1, ...]
    """
    if com.nxpt == 1:
        raise Exception("Not yet implemented for nxpt = 1")
    elif com.nxpt == 2:
        if ("snowflake45" in str(com.geometry[0])) or (
            "snowflake75" in str(com.geometry[0])
        ):
            R_xpt1 = com.rm[com.ixpt1[0], com.iysptrx1[0], 4]
            Z_xpt1 = com.zm[com.ixpt1[0], com.iysptrx1[0], 4]
            R_xpt2 = com.rm[com.ixpt1[1], com.iysptrx2[1], 4]
            Z_xpt2 = com.zm[com.ixpt1[1], com.iysptrx2[1], 4]
            R_xpt = [R_xpt1, R_xpt2]
            Z_xpt = [Z_xpt1, Z_xpt2]
        elif "snowflake135" in str(com.geometry[0]):
            R_xpt2 = com.rm[com.ixpt1[0], com.iysptrx1[0], 4]
            Z_xpt2 = com.zm[com.ixpt1[0], com.iysptrx1[0], 4]
            R_xpt1 = com.rm[com.ixpt1[1], com.iysptrx2[1], 2]
            Z_xpt1 = com.zm[com.ixpt1[1], com.iysptrx2[1], 2]
            R_xpt = [R_xpt1, R_xpt2]
            Z_xpt = [Z_xpt1, Z_xpt2]
        else:
            raise Exception("Not yet implemented for other geometries")

    return R_xpt, Z_xpt


def calc_forcebalance(bbb, com, prefix="forcebalance"):
    """
    Calculate impurity force-balance terms using UEDGE (bbb) variables.

    Parameters
    ----------
    bbb : object
        UEDGE bbb module (Fortran-Python interface)
    cutlo : float
        Small number to avoid division by zero

    Returns
    -------
    forcebalance : dict
        Dictionary containing impurity velocity and force components
    """

    # Load UEDGE variables
    # -----------------------------
    ev = bbb.ev
    misotope = bbb.misotope
    natomic = bbb.natomic
    mi = bbb.mi
    zi = bbb.zi
    rrv = com.rrv
    gpex = bbb.gpex
    gpix = bbb.gpix
    gtex = bbb.gtex
    gtix = bbb.gtix
    pri = bbb.pri
    qe = bbb.qe
    alfe = bbb.alfe
    betai = bbb.betai
    ex = bbb.ex
    vol = com.vol
    up = bbb.up
    volmsor = bbb.volmsor
    loglambda = bbb.loglambda
    pondomfpari_use = bbb.pondomfpari_use

    fricflf = bbb.fricflf
    cftaud = bbb.cftaud
    zi_in = bbb.ziin
    is_z0_imp_const = bbb.is_z0_imp_const
    z0_imp_const = getattr(bbb, "z0_imp_const", 0.0)

    ni = bbb.ni
    ne = bbb.ne
    te = bbb.te
    ti = bbb.ti
    zeff = bbb.zeff
    cutlo = com.cutlo

    # -----------------------------
    # Array sizes
    # -----------------------------
    shape = ne.shape
    nchstate = bbb.nchstate
    nion = np.sum(natomic)

    # -----------------------------
    # Allocate arrays
    # -----------------------------
    den = np.zeros((misotope, nchstate) + shape)
    gradt = np.zeros_like(den)
    gradp = np.zeros_like(den)

    upi = np.zeros(shape + (nion,))
    upi_gradp = np.zeros_like(upi)
    upi_alfe = np.zeros_like(upi)
    upi_betai = np.zeros_like(upi)
    upi_ex = np.zeros_like(upi)
    upi_volmsor = np.zeros_like(upi)
    taudeff = np.zeros_like(upi)

    F_drag = np.zeros_like(upi)
    F_thermal = np.zeros_like(upi)
    F_gradp = np.zeros_like(upi)
    F_pot = np.zeros_like(upi)
    nu = np.zeros_like(upi)

    # -----------------------------
    # Base plasma quantities
    # -----------------------------
    den[0, 0] = ne
    den[1, 0] = ni[:, :, 0]

    tempa = te
    tif = ti

    gradp[0, 0] = rrv * gpex
    gradt[0, 0] = rrv * gtex

    # -----------------------------
    # Flux limiter
    # -----------------------------
    ltmax = np.minimum(
        np.abs(tempa / (rrv * gtex + cutlo)),
        np.abs(tif / (rrv * gtix + cutlo)),
        np.abs(den[0, 0] * tempa / (rrv * gpex + cutlo)),
    )

    lmfpe = 1e16 * (tempa / ev) ** 2 / (den[0, 0] + cutlo)
    lmfpi = 1e16 * (tif / ev) ** 2 / (den[0, 0] + cutlo)

    ltmax = np.minimum(ltmax, np.abs((pri[:, :, 0]) / (rrv * gpix[:, :, 0] + cutlo)))

    flxlimf = 1.0 / (1.0 + fricflf * ((lmfpe + lmfpi) / (ltmax + cutlo)) ** 2)

    # -----------------------------
    # Impurity loop
    # -----------------------------
    ifld = 0
    zeffv = zeff

    for misa in range(2, misotope):
        for nz in range(natomic[misa]):
            ifld += 1

            # Skip neutrals
            if zi_in[ifld] < 1e-10:
                continue

            den[misa, nz] = ni[:, :, ifld]

            gradt[misa, nz] = rrv * gtix
            gradp[misa, nz] = rrv * gpix[:, :, ifld] - pondomfpari_use[:, :, ifld]

            if is_z0_imp_const == 0:
                z0 = den[0, 0] * zeffv / (den[1, 0] + cutlo) - 1.0
            else:
                z0 = z0_imp_const

            taud = (
                cftaud
                * 5.624e54
                * mi[0] ** 0.5
                * mi[ifld]
                * tif**1.5
                / (
                    loglambda * den[misa, nz] * zi[ifld] ** 2 * (mi[0] + mi[ifld])
                    + cutlo
                )
            )

            taudeff[:, :, ifld] = (
                flxlimf
                * taud
                * den[misa, nz]
                * (1 + 2.65 * z0)
                * (1 + 0.285 * z0)
                / (den[0, 0] * (1 + 0.24 * z0) * (1 + 0.93 * z0) + cutlo)
            )

            # -----------------------------
            # Force components
            # -----------------------------
            fac = taudeff[:, :, ifld] / mi[0]

            upi_gradp[:, :, ifld] = -gradp[misa, nz] / (den[misa, nz] + cutlo) * fac
            upi_alfe[:, :, ifld] = alfe[ifld] * gradt[0, 0] * fac
            upi_betai[:, :, ifld] = betai[ifld] * gradt[misa, nz] * fac
            upi_ex[:, :, ifld] = qe * zi[ifld] * rrv * ex * fac
            upi_volmsor[:, :, ifld] = (
                volmsor[:, :, ifld] / (den[misa, nz] * vol + cutlo)
            ) * fac

            # -----------------------------
            # Total impurity velocity
            # -----------------------------
            upi[:, :, ifld] = (
                up[:, :, 0]
                + upi_gradp[:, :, ifld]
                + upi_alfe[:, :, ifld]
                + upi_betai[:, :, ifld]
                + upi_ex[:, :, ifld]
                + upi_volmsor[:, :, ifld]
            )
            nu[:, :, ifld] = 1 / taudeff[:, :, ifld]

            F_drag[:, :, ifld] = (
                (den[misa, nz] + cutlo)
                * mi[0]
                * nu[:, :, ifld]
                * (upi[:, :, ifld] - up[:, :, 0])
            )
            F_gradp[:, :, ifld] = -gradp[misa, nz]
            F_thermal[:, :, ifld] = (
                alfe[ifld] * gradt[0, 0] + betai[ifld] * gradt[misa, nz]
            ) * (den[misa, nz] + cutlo)
            F_pot[:, :, ifld] = qe * zi[ifld] * rrv * ex * (den[misa, nz] + cutlo)

    # -----------------------------
    # Output dictionary
    # -----------------------------
    forcebalance = {
        "upi": upi,
        "up": up[:, :, 0],
        "upi_gradp": upi_gradp,
        "upi_alfe": upi_alfe,
        "upi_betai": upi_betai,
        "upi_ex": upi_ex,
        "upi_volmsor": upi_volmsor,
        "taudeff": taudeff,
        "F_drag": F_drag,
        "F_gradp": F_gradp,
        "F_thermal": F_thermal,
        "F_pot": F_pot,
    }
    for key, val in forcebalance.items():
        np.save(f"{prefix}_{key}.npy", val)

    return forcebalance_toggle if False else forcebalance


def integrate_var_pol(var: np.ndarray):
    """Integrate an input variable along flux tubes

    :param var: 2D UEDGE variable
    :return: Integrated variable, [ny+2]
    """
    var_integrated = np.zeros(com.ny + 2)
    for iy in range(com.ny + 2):
        # if iy <= com.iysptrx1[0]:
        var_integrated[iy] = np.sum(var[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, iy])
        if com.nxpt == 2:
            var_integrated[iy] += np.sum(var[com.ixpt1[1] + 1 : com.ixpt2[1] + 1, iy])
    # else:
    #     var_integrated[iy] = np.sum(var[1:-1, iy])
    #     if com.nxpt == 2:
    #         var_integrated[iy] += np.sum(var[1:-1, iy])

    return var_integrated


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


def get_q_plates(
    c: uetools.Case | None = None,
    xaxis: str = "r",
    include_radiation: bool = True,
    project: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Get the heat flux and coordinates along the target for each strike point

    :return: [q1, q2, ...], [r1, r2, ...]
    """
    # Get heat flux and coordinate data from case/memory
    if c is None:
        bbb.plateflux()
        if include_radiation:
            q_odata = (bbb.sdrrb + bbb.sdtrb).T
            q_idata = (bbb.sdrlb + bbb.sdtlb).T
        else:
            q_odata = bbb.sdtrb.T
            q_idata = bbb.sdtlb.T
        r_idata = com.yyrb.T
        r_odata = com.yylb.T
        if com.sibdry != 0 and com.simagx != 0:
            psin = (com.psi - com.simagx) / (com.sibdry - com.simagx)
        else:
            psin = com.psi
        ixlb = com.ixlb
        ixrb = com.ixrb
        nxpt = com.nxpt
        rr = com.rr
    else:
        if include_radiation:
            q_odata = (c.get("sdrrb") + c.get("sdtrb")).T
            q_idata = (c.get("sdrlb") + c.get("sdtlb")).T
        else:
            q_odata = c.get("sdtrb").T
            q_idata = c.get("sdtlb").T
        r_idata = c.get("yyrb").T
        r_odata = c.get("yylb").T
        sibdry = c.get("sibdry")
        simagx = c.get("simagx")
        if sibdry is not None and simagx is not None:
            if sibdry != 0 and simagx != 0:
                psin = (c.get("psi") - c.get("simagx")) / (
                    c.get("sibdry") - c.get("simagx")
                )
            else:
                psin = c.get("psi")
        else:
            psin = c.get("psi")
        ixlb = c.get("ixlb")
        ixrb = c.get("ixrb")
        nxpt = c.get("nxpt")
        bpol = c.get("bpol")
        b = c.get("b")
        rr = 0.25 * (
            bpol[:, :, 1] / b[:, :, 1]
            + bpol[:, :, 2] / b[:, :, 2]
            + bpol[:, :, 3] / b[:, :, 3]
            + bpol[:, :, 4] / b[:, :, 4]
        )

    # Organise data into SPs
    if nxpt == 1:
        q1 = q_odata[0] / rr[ixrb[0] + 1] ** project
        q2 = q_idata[0] / rr[ixlb[0]] ** project
        if xaxis == "r":
            r1 = r_idata[0]
            r2 = r_odata[0]
        elif xaxis == "psi":
            r1 = psin[ixrb[0] + 1, :, 0]
            r2 = psin[ixlb[0], :, 0]
        return [q1, q2], [r1, r2]

    elif nxpt == 2:
        q1 = q_odata[0] / rr[ixrb[0] + 1] ** project
        q2 = q_idata[1] / rr[ixlb[1] + 1] ** project
        q3 = q_odata[1] / rr[ixrb[1]] ** project
        q4 = q_idata[0] / rr[ixlb[0]] ** project
        if xaxis == "r":
            r1 = r_idata[0]
            r2 = r_odata[1]
            r3 = r_idata[1]
            r4 = r_odata[0]
        elif "psi" in xaxis:
            r1 = psin[ixrb[0] + 1, :, 0]
            r2 = psin[ixlb[1] + 1, :, 0]
            r3 = psin[ixrb[1], :, 0]
            r4 = psin[ixrb[0], :, 0]
        return [q1, q2, q3, q4], [r1, r2, r3, r4]


def get_Q_plates(
    c: uetools.Case | None = None, include_radiation: bool = True, verbose: bool = True
):
    """Get the total heat flux delivered to each strike point

    :return: P1, P2, P3, P4
    """

    q, _ = get_q_plates(
        c=c, xaxis="r", include_radiation=include_radiation, project=False
    )
    if c is None:
        sxnp = com.sxnp
        ixlb = com.ixlb
        ixrb = com.ixrb
        nxpt = com.nxpt
    else:
        sxnp = c.get("sxnp")
        ixlb = c.get("ixlb")
        ixrb = c.get("ixrb")
        nxpt = c.get("nxpt")

    if nxpt == 1:
        P1 = np.sum(q[0] * sxnp[ixrb[0]])
        P2 = np.sum(q[1] * sxnp[ixlb[0]])
        if verbose:
            print("Power delivered to each target plate: ")
            Ptot = P1 + P2
            print(
                "SP1: {:.1f}".format(100 * P1 / Ptot)
                + "%, "
                + "SP2: {:.1f}".format(100 * P2 / Ptot)
                + "%, "
            )
        return P1, P2
    elif nxpt == 2:
        P1 = np.sum(q[0] * sxnp[ixrb[0]])
        P2 = np.sum(q[1] * sxnp[ixlb[1]])
        P3 = np.sum(q[2] * sxnp[ixrb[1] + 1])
        P4 = np.sum(q[3] * sxnp[ixrb[0] + 1])
        if verbose:
            print("Power delivered to each target plate: ")
            Ptot = P1 + P2 + P3 + P4
            print(
                "SP1: {:.1f}".format(100 * P1 / Ptot)
                + "%, "
                + "SP2: {:.1f}".format(100 * P2 / Ptot)
                + "%, "
                + "SP3: {:.1f}".format(100 * P3 / Ptot)
                + "%, "
                + "SP4: {:.1f}".format(100 * P4 / Ptot)
                + "%, "
            )

        return P1, P2, P3, P4

    # if include_radiation:
    #     q_odata = (bbb.sdrrb + bbb.sdtrb).T
    #     q_idata = (bbb.sdrlb + bbb.sdtlb).T
    # else:
    #     q_odata = bbb.sdtrb.T
    #     q_idata = bbb.sdtlb.T
    # q1 = q_odata[0]
    # q2 = q_idata[1]
    # q3 = q_odata[1]
    # q4 = q_idata[0]
    # r1 = com.yyrb.T[0]
    # r2 = com.yylb.T[1]
    # r3 = com.yyrb.T[1]
    # r4 = com.yylb.T[0]
    # dr1 = get_dr_plate(r1)
    # dr2 = get_dr_plate(r2)
    # dr3 = get_dr_plate(r3)
    # dr4 = get_dr_plate(r4)

    # P1 = np.sum(q1[1:-1] * dr1[1:-1])
    # P2 = np.sum(q2[1:-1] * dr2[1:-1])
    # P3 = np.sum(q3[1:-1] * dr3[1:-1])
    # P4 = np.sum(q4[1:-1] * dr4[1:-1])

    # return P1, P2, P3, P4


def get_q_drifts():
    """Get the ExB and grad B convective heat fluxes. Outputs have dimensions [com.nx+2,com.ny+2,2], where the third dimension contains the x and y components of the vector (in UEDGE coordinates)

    :return: q_ExB, q_gradB
    """

    # q_ExB = np.zeros((com.nx + 2, com.ny + 2, 2))
    # q_gradB = np.zeros((com.nx + 2, com.ny + 2, 2))
    # p = (bbb.ne * bbb.te) + (bbb.ni[:, :, 0] * bbb.ti)
    # q_ExB[:, :, 0] = (
    #     -(5 / 2) * np.sign(bbb.b0) * np.sqrt(1 - com.rr**2) * bbb.v2ce[:, :, 0] * p
    # )
    # q_ExB[:, :, 1] = (5 / 2) * bbb.vyce[:, :, 0] * p
    # q_gradB[:, :, 0] = (
    #     -(5 / 2) * np.sign(bbb.b0) * np.sqrt(1 - com.rr**2) * bbb.v2cb[:, :, 0] * p
    # )
    # q_gradB[:, :, 1] = (5 / 2) * bbb.vycb[:, :, 0] * p

    # Compute the heat fluxes
    niy_upwind = np.where(bbb.vy[:, :, 0] > 0, bbb.niy0[:, :, 0], bbb.niy1[:, :, 0])
    ney_upwind = np.where(bbb.vy[:, :, 0] > 0, bbb.ney0[:, :], bbb.ney1[:, :])
    tey_upwind = np.where(bbb.vy[:, :, 0] > 0, bbb.tey0[:, :], bbb.tey1[:, :])
    tiy_upwind = np.where(bbb.vy[:, :, 0] > 0, bbb.tiy0[:, :], bbb.tiy1[:, :])
    ion_energy_upwindy = (
        (5 / 2) * tiy_upwind
        + (1 / 2) * bbb.mp * bbb.minu[0] * (bbb.up[:, :, 0] + bbb.vy[:, :, 0]) ** 2
    ) * niy_upwind
    electron_energy_upwindy = (5 / 2) * tey_upwind * ney_upwind
    ion_energy_x = (
        (5 / 2) * bbb.ti
        + (1 / 2) * bbb.mp * bbb.minu[0] * (bbb.up[:, :, 0] + bbb.vy[:, :, 0]) ** 2
    ) * bbb.ni[:, :, 0]
    electron_energy_x = (5 / 2) * bbb.te * bbb.ne
    q_ExB = np.zeros((com.nx + 2, com.ny + 2, 2))
    q_gradB = np.zeros((com.nx + 2, com.ny + 2, 2))
    q_ExB[:, :, 0] = (
        -np.sign(bbb.b0)
        * np.sqrt(1 - com.rr**2)
        * bbb.cf2ef
        * bbb.v2ce[:, :, 0]
        * (ion_energy_x + electron_energy_x)
    )
    q_ExB[:, :, 1] = (
        bbb.cfyef * bbb.vyce[:, :, 0] * (ion_energy_upwindy + electron_energy_upwindy)
    )
    q_gradB[:, :, 0] = (
        -np.sign(bbb.b0)
        * np.sqrt(1 - com.rr**2)
        * bbb.cf2bf
        * bbb.v2cb[:, :, 0]
        * (ion_energy_x + electron_energy_x)
    )
    q_gradB[:, :, 1] = (
        bbb.cfybf * bbb.vycb[:, :, 0] * (ion_energy_upwindy + electron_energy_upwindy)
    )

    return q_ExB, q_gradB


def get_fni_drifts(index: int = 0):
    """Get the ExB and grad B convective ion particle fluxes. Outputs have dimensions [com.nx+2,com.ny+2,2], where the third dimension contains the x and y components of the vector (in UEDGE coordinates)

    :return: fni_ExB, fni_gradB
    """

    # Compute the particle fluxes
    niy_upwind = np.where(
        bbb.vy[:, :, index] > 0, bbb.niy0[:, :, index], bbb.niy1[:, :, index]
    )

    fni_ExB = np.zeros((com.nx + 2, com.ny + 2, 2))
    fni_gradB = np.zeros((com.nx + 2, com.ny + 2, 2))
    fni_ExB[:, :, 0] = (
        -np.sign(bbb.b0)
        * np.sqrt(1 - com.rr**2)
        * bbb.cf2ef
        * bbb.v2ce[:, :, index]
        * niy_upwind[:, :]
        * com.sy[:, :]
    )
    fni_ExB[:, :, 1] = (
        bbb.cfyef * bbb.vyce[:, :, index] * niy_upwind[:, :] * com.sy[:, :]
    )
    fni_gradB[:, :, 0] = (
        -np.sign(bbb.b0)
        * np.sqrt(1 - com.rr**2)
        * bbb.cf2bf
        * bbb.v2cb[:, :, index]
        * niy_upwind[:, :]
        * com.sy[:, :]
    )
    fni_gradB[:, :, 1] = bbb.cfybf * bbb.fniycb[:, :, index]

    return fni_ExB, fni_gradB


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


def midplane_exp_fit():
    """Fit exponetial decay curve to the outer midplane electron temperature and density profiles

    :return: T_fit, lambda_T_mm, n_fit, lambda_n_mm
    """
    y = com.yyc[com.iysptrx1[0] :]
    Te = bbb.te[bbb.ixmp, com.iysptrx1[0] :] / bbb.ev
    ne = bbb.ne[bbb.ixmp, com.iysptrx1[0] :]

    expfun = lambda x, A, lamda_inv: A * np.exp(-x * lamda_inv)
    lambda_T_fit, _ = curve_fit(
        expfun,
        y,
        Te,
        p0=[np.max(Te), 100.0],
        bounds=(0, np.inf),
    )
    lambda_T_mm = 1000 / lambda_T_fit[1]
    print("lambda_T  = {:.2f} mm".format(lambda_T_mm))

    lambda_n_fit, _ = curve_fit(
        expfun,
        y,
        ne,
        p0=[np.max(ne), 100.0],
        bounds=(0, np.inf),
    )
    lambda_n_mm = 1000 / lambda_n_fit[1]
    print("lambda_n  = {:.2f} mm".format(lambda_n_mm))

    return expfun(y, *lambda_T_fit), lambda_T_mm, expfun(y, *lambda_n_fit), lambda_n_mm


def eich_exp_shahinul_odiv_final(
    omp: bool = False,
    ixmp: int = None,
    save_prefix="lambdaq_result",
    SP=1,
    ix_SP=None,
    include_radiation=True,
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

    # === Select which q_parallel to fit (OMP vs ODIV)
    bbb.plateflux()
    ppar = (
        bbb.feex + bbb.feix + 0.5 * bbb.mi[0] * bbb.up[:, :, 0] ** 2 * bbb.fnix[:, :, 0]
    )
    rrf = getrrf()
    # if ix_SP is None:
    #     if include_radiation:
    #         q_data = bbb.sdrrb + bbb.sdtrb
    #         q_ldata = bbb.sdrlb + bbb.sdtlb
    #     else:
    #         q_data = bbb.sdtrb
    #         q_ldata = bbb.sdtlb
    #     if "snowflake" in str(com.geometry):
    #         if SP == 1:
    #             q_perp_odiv = q_data[:, 0].reshape(-1)[:-1]
    #         elif SP == 2:
    #             q_perp_odiv = q_ldata[:, 1].reshape(-1)[:-1]
    #         elif SP == 3:
    #             q_perp_odiv = q_data[:, 1].reshape(-1)[:-1]
    #         elif SP == 4:
    #             q_perp_odiv = q_ldata[:, 0].reshape(-1)[:-1]
    #     else:
    #         if SP == 1:
    #             q_perp_odiv = q_data.reshape(-1)[:-1]
    #         elif SP == 2:
    #             q_perp_odiv = q_ldata.reshape(-1)[:-1]
    # else:
    #     q_para_odiv = ppar[ix_SP, :-1] / com.sx[ix_SP, :-1] / rrf[ix_SP, :-1]
    #     q_perp_odiv = q_para_odiv

    # q_para_omp = ppar[ixmp, :-1] / com.sx[ixmp, :-1] / rrf[ixmp, :-1]
    # s_omp = com.yyrb[:-1]
    # q_fit = q_para_omp if omp else q_para_odiv

    # s_omp = s_omp.flatten()
    # q_fit = q_fit.flatten()

    # interp_fun = interp1d(s_omp, q_fit, kind="cubic", fill_value="extrapolate")
    # s_interp = np.linspace(s_omp.min(), s_omp.max(), 300)
    # q_interp = interp_fun(s_interp)

    # Get heat flux at target
    if ix_SP is None:
        if include_radiation:
            q_data = bbb.sdrrb + bbb.sdtrb
            q_ldata = bbb.sdrlb + bbb.sdtlb
        else:
            q_data = bbb.sdtrb
            q_ldata = bbb.sdtlb
        if "snowflake" in str(com.geometry):
            if SP == 1:
                ix_SP = com.ixrb[0]
                q_perp_odiv = q_data[:, 0].reshape(-1)[:-1] / rrf[ix_SP + 1, :-1]
            elif SP == 2:
                ix_SP = com.ixlb[1]
                q_perp_odiv = q_ldata[:, 1].reshape(-1)[:-1] / rrf[ix_SP + 1, :-1]
            elif SP == 3:
                ix_SP = com.ixrb[1] + 1
                q_perp_odiv = q_data[:, 1].reshape(-1)[:-1] / rrf[ix_SP - 1, :-1]
            elif SP == 4:
                ix_SP = com.ixlb[0] + 1
                q_perp_odiv = q_ldata[:, 0].reshape(-1)[:-1] / rrf[ix_SP - 1, :-1]
        else:
            if SP == 1:
                ix_SP = com.ixrb[0]
                q_perp_odiv = q_data.reshape(-1)[:-1] / rrf[ix_SP + 1, :-1]
            elif SP == 2:
                ix_SP = com.ixlb[0] + 1
                q_perp_odiv = q_ldata.reshape(-1)[:-1] / rrf[ix_SP - 1, :-1]
    else:
        q_perp_odiv = ppar[ix_SP, :-1] / com.sx[ix_SP, :-1]  # / rrf[ix_SP, :-1]
    q_para_odiv = ppar[ix_SP, :-1] / com.sx[ix_SP, :-1] / rrf[ix_SP, :-1]

    # Get heat flux at midplane
    q_para_omp = ppar[ixmp, :-1] / com.sx[ixmp, :-1] / rrf[ixmp, :-1]

    q_fit = q_para_omp if omp else q_para_odiv

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
        if SP == 1:
            yyrb = com.yyrb[:, 0].reshape(-1)[:-1]
        elif SP == 2:
            yyrb = com.yylb[:, 1].reshape(-1)[:-1]
        elif SP == 3:
            yyrb = com.yyrb[:, 1].reshape(-1)[:-1]
        elif SP == 4:
            yyrb = com.yylb[:, 0].reshape(-1)[:-1]
    else:
        if SP == 1:
            yyrb = com.yyrb.reshape(-1)[:-1]
        elif SP == 2:
            yyrb = com.yylb.reshape(-1)[:-1]
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
        [0.02, 0.04, 1e9, s_fit.max() + 0.01],
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
    f: np.ndarray,
    rm: np.ndarray = None,
    zm: np.ndarray = None,
    ixmp: int = None,
    ixmp_is_above_mp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the values of an array at the midplane

    :param f: Variable on UEDGE grid
    :param rm: R grid coords, defaults to None
    :param zm: Z grid coords, defaults to None
    :param ixmp: Midplane index, defaults to None
    :param ixmp_is_above_mp: Whether cell ixmp is above the midplane or not, defaults to False
    :return: x_mp (midplane radial coordinates), f_mp (midplane radial profile)
    """
    if rm is None:
        rm = com.rm
        zm = com.zm
    if ixmp is None:
        ixmp = bbb.ixmp
    if ixmp_is_above_mp:
        x_mp = 0.5 * (rm[ixmp, :, 2] + rm[ixmp, :, 4])
        y_mp = 0.5 * (zm[ixmp, :, 4] + zm[ixmp, :, 4])
    else:
        x_mp = 0.5 * (rm[ixmp, :, 1] + rm[ixmp, :, 3])
        y_mp = 0.5 * (zm[ixmp, :, 1] + zm[ixmp, :, 3])
    if ixmp_is_above_mp:
        x_1 = rm[ixmp, :, 0]
        y_1 = zm[ixmp, :, 0]
        x_2 = rm[ixmp + 1, :, 0]
        y_2 = zm[ixmp + 1, :, 0]
    else:
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


def Psol():
    """Calculate the power crossing the separatrix

    :return: P_sol in W
    """
    # ix_mask = com.isixcore == True
    # if com.nxpt == 1:
    #     P_sol = np.sum(
    #         bbb.feey[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, com.iysptrx]
    #         + bbb.feiy[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, com.iysptrx]
    #     )
    # else:
    #     if "snowflake15" in str(com.geometry):
    #         P_sol = np.sum(
    #             bbb.feey[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, com.iysptrx]
    #             + bbb.feiy[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, com.iysptrx]
    #         )
    #         P_sol += np.sum(
    #             bbb.feey[com.ixpt1[1] + 1 : com.ixpt2[1] + 1, com.iysptrx]
    #             + bbb.feiy[com.ixpt1[1] + 1 : com.ixpt2[1] + 1, com.iysptrx]
    #         )
    #     elif "snowflake75" in str(com.geometry):
    #         P_sol = np.sum(
    #             bbb.feey[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, com.iysptrx1[0]]
    #             + bbb.feiy[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, com.iysptrx1[0]]
    #         )

    # TODO: In SF135 geometry, should be summing on iy=com.iysptrx2[0] - double check for all geometries
    ix_mask = com.isixcore == True
    P_sol = np.sum(
        bbb.feey[ix_mask, com.iysptrx1[0]] + bbb.feiy[ix_mask, com.iysptrx1[0]]
    )

    if com.isudsym == 1:
        P_sol *= 2

    return P_sol
