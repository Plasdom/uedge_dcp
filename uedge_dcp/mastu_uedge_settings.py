from uedge import *
from uedge.rundt import rundt
import os
from uedge.hdf5 import *
from scipy.interpolate import interp1d
import subprocess
import pickle
import yaml


def set_geometry(
    gridfile: str,
    geometry: str = "snull           ",
    nxpt: int = 1,
    isudsym: int = 0,
    nxleg: int = None,
    nxcore: int = None,
    ixmp: int = None,
):
    """Set the geometry

    :param gridfile: Gridfile to use
    :param geometry: Geometry, defaults to "snull           "
    :param nxpt: Number of X-points, defaults to 1
    :param isudsym: Up/down symmetry, defaults to 0
    :param nxleg: Number of pol cells in leg, defaults to None
    :param nxcore: Number of pol cells in HFS core, defaults to None
    :param ixmp: Poloidal index of outer midplane cells, defaults to None
    """
    os.system("ln -sf " + gridfile + " gridue")
    bbb.mhdgeo = 1
    com.geometry[0] = geometry
    com.isudsym = isudsym
    bbb.gengrid = 0
    bbb.newgeo = 1
    com.nxpt = nxpt
    com.isnonog = 1
    if nxleg is not None:
        com.nxleg[0, 0] = nxleg
    if nxcore is not None:
        com.nxcore[0, 0] = nxcore
    if ixmp is not None:
        bbb.ixmp = ixmp


# def set_apdirs(uebasedir: str = "/Users/power8/Documents/01_code/01_uedge/uedge"):
def set_apdirs(uebasedir: str | None = None):
    """Set directories for hydrogen and impurity data

    :param uebasedir: UEDGE source code base directory, defaults to "/Users/power8/Documents/01_code/01_uedge"
    """
    if uebasedir is None:
        hostname = subprocess.check_output(["hostname"], text=True).strip("\n")
        if hostname == "skywalker17":
            uebasedir = "/Users/power8/Documents/01_code/01_uedge/uedge"
        else:
            uebasedir = "/global/homes/d/dpow1/01_code/UEDGE"
    api.apidir = os.path.join(uebasedir, "api")
    aph.aphdir = os.path.join(uebasedir, "aph")


def set_fd_algos():
    """Set the finite difference algorithm settings"""
    # Finite-difference algorithms (upwind, central diff, etc.)
    bbb.methn = 33  # ion continuty eqn
    bbb.methu = 33  # ion parallel momentum eqn
    bbb.methe = 33  # electron energy eqn
    bbb.methi = 33  # ion energy eqn
    bbb.methg = 66  # neutral gas continuity eqn
    # allocate space for save variables
    bbb.allocate()  # allocates storage for arrays


def set_bcs(
    iflcore: int = 1,
    tcore: float = 200.0,
    pcore: float = 0.625e6,
    lyni: float = 0.1,
    lyt: float = 0.1,
    ncore: float = 3e19,
    recycw: float = 0.98,
    recycp: float = 0.98,
):
    """Set the boundary conditions

    :param iflcore: iflcore setting (0 = fixed core temperature, 1 = fixed core input power), defaults to 1
    :param tcore: Core temperature [eV] to use if iflcore = 0, defaults to 200.0
    :param pcore: Input power across core boundary if iflcore = 1, defaults to 0.625e6
    :param lyni: Density radial scale length [m], defaults to 0.1
    :param lyt: Temperature radial scale length [m], defaults to 0.1
    :param ncore: Core density, defaults to 3e19
    """
    # Boundary conditions
    bbb.isnicore[0] = 1  # =3 gives uniform density and I=curcore
    bbb.ncore[0] = ncore  # hydrogen ion density on core
    bbb.isybdryog = 1  # =1 uses orthog diff at iy=0 and iy=ny bndry

    bbb.iflcore = iflcore  # flag; =0, fixed Te,i; =1, fixed power on core
    bbb.tcoree = tcore  # core Te if iflcore=0
    bbb.tcorei = tcore  # core Ti if iflcore=0
    bbb.pcoree = pcore  # core elec power if iflcore=1
    bbb.pcorei = pcore  # core ion  power if iflcore=1

    bbb.isnwcono = 3  # =3 for (1/n)dn/dy = 1/lyni
    bbb.nwomin[0] = 1.0e13  # 1.e14 # 1.e12 #
    bbb.nwimin[0] = 1.0e13  # 1.e14 # 1.e12 #
    bbb.isnwconi = 3  # switch for private-flux wall
    # bbb.lyni[1] = 0.02		#iy=ny+1 density radial scale length (m)
    # bbb.lyni[0] = 0.02		#iy=0 density radial scale length
    bbb.lyni[1] = lyni  # iy=ny+1 density radial scale length (m)
    bbb.lyni[0] = lyni  # iy=0 density radial scale length

    bbb.istepfc = 3  # =3 for (1/Te)dTe/dy=1/lyte on pf wall
    bbb.istipfc = 3  # =3 ditto for Ti on pf wall
    bbb.istewc = 3  # =3 ditto for Te on vessel wall
    bbb.istiwc = 3  # =3 ditto for Ti on vessel wall
    # bbb.lyte = 0.01 # 0.02	# scale length for Te bc
    # bbb.lyti = 0.01 # 0.02	# scale length for Ti bc
    bbb.lyte = lyt  # 0.02	# scale length for Te bc
    bbb.lyti = lyt  # 0.02	# scale length for Ti bc
    bbb.tedge = 2.0  # fixed wall,pf Te,i if istewc=1, etc.

    bbb.isupcore = 1  # =1 sets d(up)/dy=0
    bbb.isupwo = 2  # =2 sets d(up)/dy=0
    bbb.isupwi = 2  # =2 sets d(up)/dy=0
    bbb.isplflxl = 0  # 1		#=0 for no flux limit (te & ti) at plate
    bbb.isngcore[0] = 0  # set neutral density gradient at core bndry

    bbb.matwso[0] = 1  # =1 --> make the outer wall recycling.
    bbb.matwsi[0] = 1  # =1 --> make the inner wall recycling.
    bbb.recycp = 1
    bbb.recycw = 1
    bbb.recycw[0] = recycw  # recycling coeff. at wall
    bbb.recycp[0] = recycp  # hydrogen recycling coeff. at plates
    bbb.recycm = -0.7  # mom BC at plates:up(,,2) = -recycm*up(,,1)


def set_flux_limits():
    """Set flux limits"""
    # Flux limits
    bbb.flalfe = 1.0  # electron parallel thermal conduct. coeff
    bbb.flalfi = 1.0  # ion parallel thermal conduct. coeff
    bbb.flalfv = 0.5  # ion parallel viscosity coeff
    bbb.flalfgx = 1.0  # neut. gas in poloidal direction
    bbb.flalfgy = 1.0  # neut. gas in radial direction
    bbb.flalfgxy = 1.0
    bbb.lgmax = 2e-1  # maximum scale-length for gas diffusion
    bbb.lgtmax = 2e-1  # maximum scale-length for temperature
    bbb.lgvmax = 2e-1  # maximum scale-length for viscosity
    bbb.flalftgx = 1.0  # limit x thermal transport
    bbb.flalftgy = 1.0  # limit y thermal transport
    bbb.flalfvgx = 1.0  # limit x neut visc transport
    bbb.flalfvgy = 1.0  # limit y neut visc transport
    bbb.flalfvgxy = 1.0  # limit x-y nonorthog neut visc transport

    bbb.isplflxlv = 1  # =0, flalfv not active at ix=0 & nx;=1 active all ix
    bbb.isplflxlgx = 1  # =0, flalfgx not active at ix=0 & nx;=1 active all ix
    bbb.isplflxlgxy = 1  # =0, flalfgxy not active at ix=0 & nx;=1 active all ixv
    bbb.isplflxlvgx = 1  # =0, flalfvgx not active at ix=0 & nx;=1 active all ix
    bbb.isplflxlvgxy = 1  # =0, flalfvgxy not active at ix=0 & nx;=1 active all ix
    bbb.iswflxlvgy = 1  # =0, flalfvgy not active at iy=0 & ny;=1 active all iy
    bbb.isplflxltgx = 1  # =0, flalfvgx not active at ix=0 & nx;=1 active all ix
    bbb.isplflxltgxy = 1  # =0, flalfvgxy not active at ix=0 & nx;=1 active all ix
    bbb.iswflxltgy = 1  # =0, flalfvgy not active at iy=0 & ny;=1 active all iy

    bbb.islnlamcon = 1  # =1,  The Coulomb logarithm is set to a constant value
    bbb.lnlam = 12  # Constant value used for the Coulomb logarythm

    bbb.kxe = 1.0  # elec thermal conduc scale factor;now default
    bbb.lmfplim = 1.0e3  # elec thermal conduc reduc 1/(1+mfp/lmfplim)


def set_solver():
    """Set the solver options"""
    # Solver package
    bbb.svrpkg = "nksol"  # Newton solver using Krylov method
    bbb.premeth = "ilut"  # "banded"	#Solution method for precond. Jacobian matrix

    # bbb.mfnksol = 3  # default is -3
    # bbb.epscon1 = 0.005  # default is 0.1
    # bbb.iscolnorm = 3  # default, OK
    # bbb.ftol = 1.0e-9  # default is 1.e-10
    # bbb.rlx = 0.4  # default, OK
    # bbb.delpy = 1.0e-9  # default is 1.e-7
    # bbb.del2nksol = 1.0e-14  # default, OK
    # bbb.premeth = "ilut"  # default is "ilut"
    # bbb.lenpfac = 60  # default, OK
    # bbb.lfililut = 100  # default is 50
    # bbb.ismmaxuc = 1  # default, OK
    # bbb.mmaxu = 80  # default is 25

    # bbb.svrpkg = "nksol"
    # bbb.premeth = "ilut"
    # bbb.runtim = 1.0e-07  # convergence for VODPK
    # bbb.rwmin = 1.0e-11  # convergence for NEWTON
    # bbb.ftol = 1.0e-8  # convergence for NKSOL
    # bbb.iscolnorm = 3
    # bbb.mfnksol = 3
    # bbb.scrit = 1.0e-3
    # bbb.nmaxnewt = 20
    # bbb.itermx = 30
    # # Jacobian increments --
    # bbb.xrinc = 2
    # bbb.xlinc = 3
    # bbb.yinc = 2
    # bbb.delpy = 1.0e-07
    # # vodpk parameters --
    # bbb.jacflg = 1
    # bbb.jpre = 1
    # bbb.rtolv = 1.0e-4


def set_h_gas(fluid_neuts: bool = True):
    """Apply neutral hydrogen settings"""
    # Atomic Physics
    com.istabon = 10  # 10=>Stotler tables
    bbb.isrecmon = 1  # =1 for recombination

    # Neutral gas properties
    com.ngsp = 1
    bbb.ineudif = 2  # pressure driven neutral diffusion
    bbb.ngbackg = 1.0e13  # 1.e15 # 1.e12 #          # background gas level (1/m**3)
    if fluid_neuts:
        bbb.isupgon[0] = 1
        bbb.isngon[0] = 0
        com.nhsp = 2
        bbb.ziin[com.nhsp - 1] = 0
        bbb.gcfacgx = 1.0  # sets plate convective gas flux
        bbb.gcfacgy = 1.0  # sets wall convective gas flux
    else:
        bbb.isupgon[0] = 0
        bbb.isngon[0] = 1
        com.nhsp = 1


def set_carbon_imps(fluid_neuts: bool = False, evolve_mom_eqs: bool = False):
    """Apply carbon impurity settings"""
    # Impurities
    bbb.isimpon = 6
    com.ngsp = 2
    com.nzsp[0] = 6  # NUMBER OF IMPURITY SPECIES FOR CARBON

    # CARBON

    bbb.isngon[1] = 1  # turns on impurity gas
    bbb.n0g[1] = 1.0e17
    if fluid_neuts:
        bbb.isupgon[1] = 1  # impurity gas is diffusive
    else:
        bbb.isupgon[1] = 0  # impurity gas is diffusive
    bbb.recycp[1] = 1.0e-10  # recycling of impurity species
    bbb.recycw[1] = 1.0e-10  # recycling at wall, do not set =0!
    bbb.ngbackg[1] = 1.0e10  # background density for impurity gas

    # Carbon ion species
    bbb.allocate()  # allocate space for source arrays,
    # and also ni and up for impurity species.
    bbb.minu[com.nhsp : com.nhsp + 6] = 12.0  # mass in AMU
    bbb.ziin[com.nhsp : com.nhsp + 6] = array(
        [1, 2, 3, 4, 5, 6]
    )  # iota(6)	# charge of each impurity species
    bbb.znuclin[0 : com.nhsp] = 1  # nuclear charge
    bbb.znuclin[com.nhsp : com.nhsp + 6] = 6  # nuclear charge of impurities
    bbb.nzbackg = 1.0e10  # background density for impurities
    bbb.n0[com.nhsp : com.nhsp + 6] = 1.0e17  # global density normalization
    bbb.inzb = 2  # parameter for implementing impurity floor
    bbb.isbohmms = 0  # Bohm BC at plates for impurities
    bbb.isnicore[com.nhsp : com.nhsp + 6] = 0  # =0 for core flux BC =curcore
    # =1 for fixed core density BC
    # =3 constant ni on core,
    #           total flux=curcore
    if evolve_mom_eqs:
        bbb.nusp_imp = 6
    bbb.recycc[1] = 0  # no core recycling of carbon gas

    bbb.curcore[com.nhsp : com.nhsp + 6] = 0.0

    bbb.isnwcono[com.nhsp : com.nhsp + 6] = (
        3  # 1 - set to nwallo ; 3 - set to (1/n)dn/dy = 1/lyni
    )
    bbb.nwallo = 1.0e12
    bbb.isnwconi[com.nhsp : com.nhsp + 6] = 3
    bbb.nwimin[com.nhsp : com.nhsp + 6] = 1.0e7
    bbb.nwomin[com.nhsp : com.nhsp + 6] = 1.0e7

    bbb.rcxighg = 0.0  # force charge exchange small
    bbb.kelighi[bbb.iigsp - 1] = 5.0e-16  # sigma-v for elastic collisions
    bbb.kelighg[bbb.iigsp - 1] = 5.0e-16
    com.iscxfit = 2  # use reduced Maggi CX fits

    # Impurity data files
    bbb.ismctab = 2  # =1 use multi charge tables from INEL
    # =2 Use multi charge tables from Strahl
    com.nzdf = 1
    com.mcfilename = ["b2frates_C"]  # , "b2frates_Li_v4"]
    com.isrtndep = 1


def set_carbon_sputtering(fhaasz: float = 0):
    """Apply carbon sputtering settings

    :param fhaasz: Coefficient for chemical and physical sputtering. When fhaasz=1 usual Haasz model is used, but convergence is
    reached more easily by steadily increasing this value from 0, defaults to 0
    """
    # Sputtering
    bbb.isch_sput[1] = 7  # Haasz/Davis sputtering model
    bbb.isph_sput[1] = 3  # Haasz/Davis sputtering model
    bbb.t_wall = 300
    bbb.t_plat = 600
    bbb.crmb = bbb.minu[0]  # set mass of bombarding particles

    bbb.fchemylb = fhaasz
    bbb.fchemyrb = fhaasz
    bbb.fphysylb = fhaasz
    bbb.fphysyrb = fhaasz
    bbb.fchemygwi = fhaasz
    bbb.fchemygwo = fhaasz


def set_transport_coeffs_DM(
    k_x: list = [-0.025, -0.02, -0.0025, 0.0004],
    k_v: list = [2.5, 0.1, 0.1, 10.0],
    d_x: list = [-0.025, -0.01, 0.0004, 0.01],
    d_v: list = [2, 0.1, 0.1, 0.5],
    inplace: bool = False,
    dif_div: float = None,
    ky_div: float = None,
    travis: float = 1,
    reverse_r_mp: bool = False,
):
    """Set radially-dependent transport coefficients

    :param k_x: Radial coordinates (x-axis) of piecewise-linear function for thermal transport coeffs vs. radial location, defaults to [-0.025, -0.02, -0.0025, 0.0004]
    :param k_v: Values (y-axis) of piecewise-linear function for thermal transport coeffs vs. radial location, defaults to [2.5, 0.1, 0.1, 10.0]
    :param d_x: Radial coordinates (x-axis) of piecewise-linear function for particle transport coeffs vs. radial location, defaults to [-0.025, -0.01, 0.0004, 0.01]
    :param d_v: Values (y-axis) of piecewise-linear function for particle transport coeffs vs. radial location, defaults to [2, 0.1, 0.1, 0.5]
    :param inplace: Whether to modify the UEDGE values in place or return as arrays, defaults to False
    :param dif_div: Value of particle transport coeffs to use below X-point, defaults to None
    :param ky_div: Value of thermal transport coeffs to use below X-point, defaults to None
    :param travis: Spatially-uniform viscosity, defaults to 1
    :return: If inplace=True, return nothing. If inplace=False, return ky, dn arrays
    """
    runid = 1
    grd.readgrid("gridue", runid)
    geometry = str(com.geometry[0])

    # Set/reset other transport coeffs before calculating D_y and K_y
    bbb.kye = 0  # 0.5		#chi_e for radial elec energy diffusion
    bbb.kyi = 0  # 0.5		#chi_i for radial ion energy diffusion
    bbb.difni[0] = 0  # .2  		#D for radial hydrogen diffusion        difniv()
    bbb.difni = 0
    bbb.travis[0] = travis  # eta_a for radial ion momentum diffusion
    bbb.difutm = 1.0  # toroidal diffusivity for potential

    # Now calculate D_y and K_y
    # Define the functional form of diffusion coeffs vs. distance from separatrix
    from scipy.interpolate import interp1d

    # k_x = [-0.025, -0.02, -0.0025, 0.0004]
    # k_v = [2.5, 0.1, 0.1, 10.0]
    k_interp_func = interp1d(k_x, k_v, fill_value=(k_v[0], k_v[-1]), bounds_error=False)

    # d_x = [-0.025, -0.01, 0.0004, 0.01]
    # d_v = [2, 0.1, 0.1, 0.5]
    d_interp_func = interp1d(d_x, d_v, fill_value=(d_v[0], d_v[-1]), bounds_error=False)

    # Find distance from separatrix for each radial cell
    r_sepx = com.rm[bbb.ixmp, com.iysptrx1[0], 3]
    r_omp = 0.5 * (com.rm[bbb.ixmp, :, 3] + com.rm[bbb.ixmp, :, 1])
    psi_sepx = com.psi[bbb.ixmp, com.iysptrx1[0], 3]
    psi_omp = 0.5 * (com.psi[bbb.ixmp, :, 3] + com.psi[bbb.ixmp, :, 1])
    if reverse_r_mp:
        dist_from_sepx = -(r_omp - r_sepx)
    else:
        dist_from_sepx = r_omp - r_sepx

    # Interpolate diffusion coeffs onto radial cells
    k_radial = k_interp_func(dist_from_sepx)
    d_radial = d_interp_func(dist_from_sepx)

    # Apply to all poloidal cells
    k_use = np.zeros((com.nx + 2, com.ny + 2))
    d_use = np.zeros((com.nx + 2, com.ny + 2))

    # Set the values in the core and SOL
    for iy in range(com.ny + 2):
        k_use[:, iy] = k_radial[iy]
        d_use[:, iy] = d_radial[iy]

        # Set the values below the X-point
        if "snowflake15" in geometry:
            for iy in range(com.ny + 2):
                if ky_div is None:
                    k_use[: com.ixpt1[0] + 1, iy] = k_radial[-1]
                    k_use[com.ixpt2[1] + 1 :, iy] = k_radial[-1]
                    k_use[com.ixpt2[0] + 1 : com.ixpt1[1] + 1, iy] = k_radial[-1]
                else:
                    k_use[: com.ixpt1[0] + 1, iy] = ky_div
                    k_use[com.ixpt2[1] + 1 :, iy] = ky_div
                    k_use[com.ixpt2[0] + 1 : com.ixpt1[1] + 1, iy] = ky_div
                if dif_div is None:
                    d_use[: com.ixpt1[0] + 1, iy] = d_radial[-1]
                    d_use[com.ixpt2[1] + 1 :, iy] = d_radial[-1]
                    d_use[com.ixpt2[0] + 1 : com.ixpt1[1] + 1, iy] = d_radial[-1]
                else:
                    d_use[: com.ixpt1[0] + 1, iy] = dif_div
                    d_use[com.ixpt2[1] + 1 :, iy] = dif_div
                    d_use[com.ixpt2[0] + 1 : com.ixpt1[1] + 1, iy] = dif_div
        else:
            for iy in range(com.ny + 2):
                if ky_div is None:
                    k_use[: com.ixpt1[0] + 1, iy] = k_radial[-1]
                    k_use[com.ixpt2[0] + 1 :, iy] = k_radial[-1]
                else:
                    k_use[: com.ixpt1[0] + 1, iy] = ky_div
                    k_use[com.ixpt2[0] + 1 :, iy] = ky_div
                if dif_div is None:
                    d_use[: com.ixpt1[0] + 1, iy] = d_radial[-1]
                    d_use[com.ixpt2[0] + 1 :, iy] = d_radial[-1]
                else:
                    d_use[: com.ixpt1[0] + 1, iy] = dif_div
                    d_use[com.ixpt2[0] + 1 :, iy] = dif_div

    # Assign calculated values in UEDGE
    target_ky = k_use
    target_dif = np.zeros(bbb.dif_use.shape)
    for isp in range(target_dif.shape[-1]):
        target_dif[:, :, isp] = d_use

    if inplace:
        bbb.kye_use = target_ky
        bbb.kyi_use = target_ky
        bbb.dif_use = target_dif
        return
    else:
        return target_ky, target_dif


def set_perp_transport_coeffs_DM_old(ky_div: float = 1.0, dif_div: float = 1.0):
    """Set the perpendicular transport coefficients according to the value shared by David Moulton for MAST-U"""
    runid = 1
    grd.readgrid("gridue", runid)

    # Set/reset other transport coeffs before calculating D_y and K_y
    bbb.kye = 0  # 0.5		#chi_e for radial elec energy diffusion
    bbb.kyi = 0  # 0.5		#chi_i for radial ion energy diffusion
    bbb.difni[0] = 0  # .2  		#D for radial hydrogen diffusion        difniv()
    bbb.difni = 0
    bbb.travis[0] = 1.0  # eta_a for radial ion momentum diffusion
    bbb.difutm = 1.0  # toroidal diffusivity for potential

    # Now calculate D_y and K_y

    # Define the functional form of diffusion coeffs vs. distance from separatrix
    k_x = [-0.025, -0.02, -0.0025, 0.0004]
    k_v = [2.5, 0.75, 0.75, 10.0]
    k_interp_func = interp1d(k_x, k_v, fill_value=(k_v[0], k_v[-1]), bounds_error=False)

    d_x = [-0.025, -0.01, 0.0004, 0.01]
    d_v = [2, 0.1, 0.1, 0.5]
    d_interp_func = interp1d(d_x, d_v, fill_value=(d_v[0], d_v[-1]), bounds_error=False)

    # Find distance from separatrix for each radial cell
    r_sepx = com.rm[bbb.ixmp, com.iysptrx1[0], 3]
    r_omp = 0.5 * (com.rm[bbb.ixmp, :, 3] + com.rm[bbb.ixmp, :, 1])
    psi_sepx = com.psi[bbb.ixmp, com.iysptrx1[0], 3]
    psi_omp = 0.5 * (com.psi[bbb.ixmp, :, 3] + com.psi[bbb.ixmp, :, 1])
    dist_from_sepx = r_sepx - r_omp

    # Interpolate diffusion coeffs onto radial cells
    k_radial = k_interp_func(dist_from_sepx)
    d_radial = d_interp_func(dist_from_sepx)

    # Apply to all poloidal cells
    k_use = np.zeros((com.nx + 2, com.ny + 2))
    d_use = np.zeros((com.nx + 2, com.ny + 2))

    # Single null case
    if com.nxpt == 1:
        # Set the values in the core and SOL
        for iy in range(com.ny + 2):
            k_use[:, iy] = k_radial[iy]
            d_use[:, iy] = d_radial[iy]

        # # Set values in the PFR
        # dist_interp_func = interp1d(psi_omp - psi_sepx, r_omp - r_sepx)
        # for ix in range(com.ixpt1[0] + 1):
        #     for iy in range(com.iysptrx1[0] + 1):
        #         psi_dist = abs(com.psi[ix, iy, 0] - psi_sepx)
        #         r_dist = dist_interp_func(psi_dist)
        #         k_use[ix, iy] = k_interp_func(r_dist)
        #         d_use[ix, iy] = d_interp_func(r_dist)
        # for ix in range(com.ixpt2[0] + 1, com.nx + 2):
        #     for iy in range(com.iysptrx1[0] + 1):
        #         psi_dist = abs(com.psi[ix, iy, 0] - psi_sepx)
        #         r_dist = dist_interp_func(psi_dist)
        #         k_use[ix, iy] = k_interp_func(r_dist)
        #         d_use[ix, iy] = d_interp_func(r_dist)

        # Set the values below the X-point
        for iy in range(com.ny + 2):
            k_use[: com.ixpt1[0] + 1, iy] = ky_div
            k_use[com.ixpt2[0] + 1 :, iy] = ky_div
            d_use[: com.ixpt1[0] + 1, iy] = dif_div
            d_use[com.ixpt2[0] + 1 :, iy] = dif_div

    # Snowflake case (need to check this works for configs other than SF75)
    elif com.nxpt == 2:
        # Set the values in the core and SOL
        for iy in range(com.ny + 2):
            k_use[:, iy] = k_radial[iy]
            d_use[:, iy] = d_radial[iy]

        # Set the values below the X-point
        for iy in range(com.ny + 2):
            k_use[: com.ixpt1[0] + 1, iy] = ky_div
            k_use[com.ixpt1[1] - 1 :, iy] = ky_div
            d_use[: com.ixpt1[0] + 1, iy] = dif_div
            d_use[com.ixpt1[1] - 1 :, iy] = dif_div

        # # Set values in the PFR
        # for ix in range(com.ixpt1[0] + 1):
        #     for iy in range(com.iysptrx1[0] + 1):
        #         k_use[ix, iy] = k_v[-1]
        #         d_use[ix, iy] = d_v[-1]
        # for ix in range(com.ixpt2[0] + 1, com.ixlb[1] + 1):
        #     for iy in range(com.iysptrx1[0] + 1):
        #         k_use[ix, iy] = k_v[-1]
        #         d_use[ix, iy] = d_v[-1]
        # for ix in range(com.ixlb[1] + 1, com.nx + 2):
        #     for iy in range(com.ny + 2):
        #         k_use[ix, iy] = k_v[-1]
        #         d_use[ix, iy] = d_v[-1]

    # Assign calculated values in UEDGE
    bbb.kye_use = k_use
    bbb.kyi_use = k_use
    for isp in range(bbb.dif_use.shape[-1]):
        bbb.dif_use[:, :, isp] = d_use


def set_perp_transport_coeffs(
    spatially_dependent: bool = False,
    kye_core: float = 2.5,
    kye_sol: float = 10.0,
    kye_div: float = None,
    kyi_core: float = 2.5,
    kyi_sol: float = 10.0,
    kyi_div: float = None,
    dif_core: float = 2.0,
    dif_sol: float = 0.5,
    dif_div: float = None,
    travis: float = 1.0,
):
    """Set perpendicular transport coefficients

    :param spatially_dependent: Whetehr to use spatially dependent coefficients (i.e. different in core and SOL), defaults to False
    :param kye_core: Value for electron thermal diffusivity in core, defaults to 2.5
    :param kye_sol: Value for electron thermal diffusivity in SOL, defaults to 10.0
    :param kye_div: Value for particle diffusivity in divertor (below X-point), defaults to be same as kye_sol
    :param kyi_core: Value for ion thermal diffusivity in core, defaults to 2.5
    :param kyi_sol: Value for ion thermal diffusivity in SOL, defaults to 10.0
    :param kyi_div: Value for particle diffusivity in divertor (below X-point), defaults to be same as kyi_sol
    :param dif_core: Value for particle diffusivity in core, defaults to 2.0
    :param dif_sol:  Value for particle diffusivity in SOL, defaults to 0.5
    :param dif_div:  Value for particle diffusivity in divertor (below X-point), defaults to be same as dif_sol
    """
    if spatially_dependent:
        # Transport coefficients
        bbb.kye = 0  # 0.5		#chi_e for radial elec energy diffusion
        bbb.kyi = 0  # 0.5		#chi_i for radial ion energy diffusion
        bbb.difni[0] = 0  # .2  		#D for radial hydrogen diffusion        difniv()
        bbb.difni = 0
        bbb.travis[0] = travis  # eta_a for radial ion momentum diffusion
        bbb.difutm = 1.0  # toroidal diffusivity for potential

        # Calculating transport coefficients

        runid = 1
        grd.readgrid("gridue", runid)

        if dif_div is None:
            for isp in range(bbb.dif_use.shape[-1]):
                bbb.dif_use[:, :, isp] = two_zone_diff_coeffs(dif_core, dif_sol)
        else:
            for isp in range(bbb.dif_use.shape[-1]):
                bbb.dif_use[:, :, isp] = three_zone_diff_coeffs(
                    dif_core, dif_sol, dif_div
                )
        if kye_div is None:
            bbb.kye_use = two_zone_diff_coeffs(kye_core, kye_sol)
        else:
            bbb.kye_use = three_zone_diff_coeffs(kye_core, kye_sol, dif_div)
        if kyi_div is None:
            bbb.kyi_use = two_zone_diff_coeffs(kyi_core, kyi_sol)
        else:
            bbb.kyi_use = three_zone_diff_coeffs(kyi_core, kyi_sol, dif_div)

    else:
        bbb.kye = 5
        bbb.kyi = 5
        bbb.difni = 1
        bbb.travis[0] = travis  # eta_a for radial ion momentum diffusion
        bbb.difutm = 1.0  # toroidal diffusivity for potential


def set_perp_transport_coeffs_ak(spatially_dependent: bool = False):
    """Set the perpendicular transport coefficients to use"""

    if spatially_dependent:

        # Transport coefficients
        bbb.kye = 0  # 0.5		#chi_e for radial elec energy diffusion
        bbb.kyi = 0  # 0.5		#chi_i for radial ion energy diffusion
        bbb.difni[0] = 0  # .2  		#D for radial hydrogen diffusion        difniv()
        bbb.difni = 0
        bbb.travis[0] = 1.0  # eta_a for radial ion momentum diffusion
        bbb.difutm = 1.0  # toroidal diffusivity for potential

        # Calculating transport coefficients

        runid = 1
        grd.readgrid("gridue", runid)

        k_x = [-0.025, -0.02, -0.0025, 0.0004]
        k_v = [2.5, 0.75, 0.75, 10.0]

        d_x = [-0.025, -0.01, 0.0004, 0.01]
        d_v = [2, 0.1, 0.1, 0.5]

        k_psi = [0.0] * (com.ny + 1)
        k_v_psi = [0.0] * (com.ny + 1)
        d_v_psi = [0.0] * (com.ny + 1)

        psi_sep = com.psi[com.ixpt2[0], com.iysptrx1[0], 4]

        for ind_j in range(0, com.ny + 1):
            if com.psi[com.nxleg[0, 0] + com.nxcore[0, 0] + 1, ind_j, 4] < psi_sep:
                bet = (
                    psi_sep
                    - com.psi[com.nxleg[0, 0] + com.nxcore[0, 0] + 1, ind_j - 1, 4]
                ) / (
                    com.psi[com.nxleg[0, 0] + com.nxcore[0, 0] + 1, ind_j, 4]
                    - com.psi[com.nxleg[0, 0] + com.nxcore[0, 0] + 1, ind_j - 1, 4]
                )
                r_sep = com.rm[
                    com.nxleg[0, 0] + com.nxcore[0, 0] + 1, ind_j - 1, 4
                ] + bet * (
                    com.rm[com.nxleg[0, 0] + com.nxcore[0, 0] + 1, ind_j, 4]
                    - com.rm[com.nxleg[0, 0] + com.nxcore[0, 0] + 1, ind_j - 1, 4]
                )
                break

        for ind_j in range(0, com.ny + 1):
            k_psi[ind_j] = com.psi[com.nxleg[0, 0] + com.nxcore[0, 0] + 1, ind_j, 4]
            dist = com.rm[com.nxleg[0, 0] + com.nxcore[0, 0] + 1, ind_j, 0] - r_sep

            for i in range(0, len(k_x)):
                if k_x[i] > dist:
                    break
            if k_x[i] <= dist:
                i = i + 1

            if i > 0:
                if i > len(k_x) - 1:
                    v = k_v[len(k_x) - 1]
                else:
                    bet = (dist - k_x[i - 1]) / (k_x[i] - k_x[i - 1])
                    v = k_v[i - 1] + (k_v[i] - k_v[i - 1]) * bet
            else:
                v = k_v[0]

            for i in range(0, len(k_x)):
                if d_x[i] > dist:
                    break

            if d_x[i] <= dist:
                i = i + 1

            if i > 0:
                if i > len(d_x) - 1:
                    d = d_v[len(k_x) - 1]
                else:
                    bet = (dist - d_x[i - 1]) / (d_x[i] - d_x[i - 1])
                    d = d_v[i - 1] + (d_v[i] - d_v[i - 1]) * bet
            else:
                d = d_v[0]

            k_v_psi[ind_j] = v
            d_v_psi[ind_j] = d

        bbb.kyi_use = k_v_psi[com.ny]
        bbb.dif_use = d_v_psi[com.ny]

        for ind_i in range(com.ixpt1[0] + 1, com.ixpt2[0] + 1):
            for ind_j in range(0, com.ny + 1):
                psi_ind = com.psi[ind_i, ind_j, 4]

                for i in range(0, com.ny + 1):
                    if k_psi[i] < psi_ind:
                        break
                if k_psi[i] >= psi_ind:
                    i = i + 1

                if i > 1:
                    if i > com.ny:
                        v = k_v_psi[com.ny]
                        d = d_v_psi[com.ny]
                    else:
                        bet = (psi_ind - k_psi[i - 1]) / (k_psi[i] - k_psi[i - 1])
                        v = k_v_psi[i - 1] + (k_v_psi[i] - k_v_psi[i - 1]) * bet
                        d = d_v_psi[i - 1] + (d_v_psi[i] - d_v_psi[i - 1]) * bet
                else:
                    v = k_v_psi[0]
                    d = d_v_psi[0]

                bbb.kyi_use[ind_i, ind_j] = v
                bbb.dif_use[ind_i, ind_j] = d

        bbb.kye_use = bbb.kyi_use

    else:
        bbb.kye = 5
        bbb.kyi = 5
        bbb.difni = 1
        bbb.travis[0] = 1.0  # eta_a for radial ion momentum diffusion
        bbb.difutm = 1.0  # toroidal diffusivity for potential


def set_div_gas_puff_h():
    """Initialise the divertor gas puff settings for hydrogen"""
    # Divertor gas puff
    bbb.nwsor = 2  # number of wall sources

    # Source 1 (setting albedo for hydrogen):
    bbb.igspsori[0] = 1  # index of gas species for inner wall sources
    bbb.igspsoro[0] = 1  # index of gas species for outer wall sources

    bbb.igasi[0] = 0.0  # amps (as if neutral has charge e)
    # of injected neutrals at inner wall source
    bbb.igaso[0] = 0.0  # amps (as if neutral has charge e)
    # of injected neutrals at outer wall source
    bbb.xgasi[0] = 0.0  # position (in m) of source
    bbb.xgaso[0] = 0.0  # position (in m) of source

    bbb.issorlb[0] = (
        1  # determines whether the source location should be computed from left of right boundary
    )
    bbb.matwso[0] = 1  # =1 --> make the outer wall recycling.
    bbb.matwsi[0] = 1  # =1 --> make the inner wall recycling.

    bbb.albdsi[0] = 1.0
    bbb.albdso[0] = 1.0


def set_div_gas_puff_carbon():
    """Initialise the divertor gas puff settings for carbon"""
    # Sources 2 (setting albedo for impurities):
    bbb.igspsori[1] = 2  # index of gas species for inner wall sources
    bbb.igspsoro[1] = 2  # index of gas species for outer wall sources

    bbb.jxsoro[1] = 1  # determines to which mesh region the source is applied to
    bbb.jxsori[1] = 1

    bbb.igasi[1] = 0.0  # amps (as if neutral has charge e)
    # of injected neutrals at inner wall source
    bbb.igaso[1] = 0.0  # amps (as if neutral has charge e)
    # of injected neutrals at outer wall source
    bbb.xgasi[1] = 0.0  # position (in m) of source
    bbb.xgaso[1] = 0.0  # position (in m) of source

    bbb.issorlb[1] = (
        1  # determines whether the source location should be computed from left of right boundary
    )
    bbb.matwso[1] = 1  # =1 --> make the outer wall recycling.
    bbb.matwsi[1] = 1  # =1 --> make the inner wall recycling.

    bbb.albdsi[1] = 1.0
    bbb.albdso[1] = 1.0

    bbb.wgasi = 1e6
    bbb.wgaso = 1e6


def set_sym_diff_xpt(isnfmiy: int = 1):
    """Apply symmetric differencing of momentum equations near X-points"""
    # Symmetric differencing of momentum equations near x-points:
    bbb.isnfmiy = isnfmiy


def set_initial_conditions():
    """Set initial values of Ti, Te, etc"""
    bbb.restart = 1
    bbb.isbcwdt = 1

    bbb.allocate()
    bbb.tes = 10 * bbb.ev
    bbb.tis = 10 * bbb.ev
    bbb.nis[:, :, 0] = 1e20
    bbb.nis[:, :, 1:] = 1e16
    bbb.ngs = 1e16
    bbb.ups = 0
    bbb.isbcwdt = 1


def set_drifts(b0_scale: float = 65.0, diamagnetic_coeff: float = 0.0):
    """Turn on drifts

    :param b0_scale: Scale factor on b0: higher values suppresses drifts. This parameter can steadily reduced to 1 to assist with convergence.
    """
    bbb.isphion = 1  # user:turns on (=1) potential eqn.

    bbb.newbcl = 1  # Sheath boundary condition (bcee, i) from current equation
    bbb.newbcr = 1  # Sheath boundary condition (bcee, i) from current equation
    bbb.b0 = b0_scale  # scale factor for magnetic field (just toroidal or total?)
    # =1 for normal direction B field
    bbb.rsigpl = 1.0e-8  # anomalous cross field conductivity / ad hoc radial electrical conductivity - global

    bbb.cfjhf = 1.0  # Coef for convective cur (fqp) heat flow
    bbb.cfjve = 1.0  # Coef for J-contribution to ve.
    bbb.cfjpy = 0  # Coef for B x gradP terms in div(J) eqn
    bbb.cfjp2 = 0  # Coef for B x gradP terms in div(J) eqn

    bbb.isfdiax = (
        1  # switch to turn on diamagnetic drift for sheath potential calculation
    )
    bbb.cfqydbo = 1  # factor to includ. fqyd in core current B.C. only
    bbb.cfydd = diamagnetic_coeff  # Coef for diamagnetic drift in y-direction
    bbb.cf2dd = diamagnetic_coeff  # Coef for diamagnetic drift in 2-direction

    bbb.cftdd = diamagnetic_coeff  # Coef for diamagnetic drift in toroidal direction
    bbb.cfyef = 1.0  # Coef for ExB drift in y-direction
    bbb.cftef = 1  # Coef for ExB drift in toroidal direction
    bbb.cf2ef = 1.0  # EXB drift in 2 direction
    bbb.cfybf = 1.0  # Coef for Grad B drift in y-direction
    bbb.cf2bf = 1  # Coef for Grad B drift in 2-direction
    bbb.cfqybf = 1  # Coef for Grad_B current in y-direction
    bbb.cfq2bf = 1  # Coef for Grad_B current in 2-direction

    bbb.cfqybbo = 0  # factor to includ. fqyb in core current B.C. only / turn off Grad B current on boundary
    bbb.cfniybbo = 0  # factor to includ. vycb in fniy,feiy at iy=0 only
    bbb.cfeeydbo = 0  # factor to includ. vycp in feey at iy=0 only

    bbb.cfniydbo = 1  # factor to includ. vycp in fniy,feiy at iy=0 only
    bbb.cfeeydbo = 1  # factor to includ. vycp in feey at iy=0 only
    bbb.cfeixdbo = 1  # factor includ v2cdi & BxgradTi in BC at ix=0,nx
    bbb.cfeexdbo = 1  # factor includ v2cde & BxgradTe in BC at ix=0,nx
    bbb.cfqym = 1  # Coef for spatial inertial rad current in y-dir.


def set_drifts_maxim(b0_scale: float = 10):
    """Turn on drifts (template taken from https://github.com/LLNL/UEDGE/blob/master/wikidocs/turn_on_drifts.html)

    :param b0_scale: Scale factor on b0: higher values suppresses drifts. This parameter can steadily reduced to 1 to assist with convergence.
    """
    bbb.isphion = 1
    bbb.b0 = b0_scale  # =1 for normal direction B field
    bbb.rsigpl = 1.0e-8  # anomalous cross field conductivity
    bbb.cfjhf = 1.0  # turn on heat flow from current (fqp)
    bbb.cfjve = 1.0  # makes vex=vix-cfjve*fqx
    bbb.jhswitch = 0  # Joule Heating switch
    bbb.newbcl = 1  # Sheath boundary condition (bcee, i) from current equation
    bbb.newbcr = 1
    bbb.isfdiax = 1  # Factor to turn on diamagnetic contribution to sheath
    bbb.cfyef = 1.0  # EXB drift in y direction
    bbb.cf2ef = 1.0  # EXB drift in 2 direction
    bbb.cfybf = 1.0  # turns on vycb - radial Grad B drift
    bbb.cf2bf = 1.0  # turns on v2cb - perp. Grad B drift (nearly pol)
    bbb.cfqybf = 1.0  # turns on vycb contrib to radial current
    bbb.cfq2bf = 1.0  # turns on v2cb contrib to perp("2") current
    bbb.cfydd = 0.0  # turns off divergence free diamagnetic current
    bbb.cf2dd = 0.0  # turns off divergence free perp diagmatic current
    bbb.cfqybbo = 0  # turn off Grad B current on boundary
    bbb.cfqydbo = 1  # use full diagmagetic current on boundary to force j_r=0
    bbb.cfniybbo = 1.0  # use to avoid artificial source at core boundary
    bbb.cfniydbo = 0.0  # use to avoid artificial source at core boundary
    bbb.cfeeybbo = 1.0  # ditto
    bbb.cfeeydbo = 0.0  # ditto
    bbb.cfeixdbo = 1.0  # turn on BXgrad(T) drift in plate BC
    bbb.cfeexdbo = 1.0  # turn on diamagnetic drift in plate BC
    bbb.cftef = 1.0  # turns on v2ce for toroidal velocity
    bbb.cftdd = 1.0  # turns on v2dd (diamag vel) for toloidal velocity
    bbb.cfqym = 1.0  # turns on inertial correction to fqy current
    bbb.iphibcc = 3  # don't set extrapolation BC for Er at iy=0
    bbb.iphibcwi = 0  # set ey=0 on inner wall if =0
    # phi(PF)=phintewi*te(ix,0) on PF wall if =1
    bbb.iphibcwo = 0  # same for outer wall
    bbb.isutcore = 2  # =1, set dut/dy=0 on iy=0 (if iphibcc=0)
    # =0, toroidal angular momentum=lzcore on iy=0 (iphibcc=0)
    bbb.isnewpot = 1
    bbb.rnewpot = 1.0
    bbb.cfnus_i = 1.0
    bbb.cfnus_e = 1.0  # include collisionality in drift effects
    bbb.isybdrywd = 1  # use diffusive flux only on y boundary
    bbb.lfililut = 200
    bbb.lenpfac = 150
    bbb.lenplufac = 150
    bbb.allocate()


def initial_short_run(dt: float = 1e-12):
    """Do an initial short run prior to calling rundt()"""
    bbb.restart = 1
    bbb.isbcwdt = 1
    bbb.dtreal = dt
    bbb.ftol = 1e-3
    bbb.icntnunk = 0
    bbb.itermx = 30
    bbb.exmain()
    bbb.itermx = 7


def add_carbon():
    """Add carbon impurities to a solution with only hydrogen. Carbon charge states (including neutrals) are initialised with a small density."""
    set_carbon_imps()
    set_carbon_sputtering(fhaasz=0.0001)
    bbb.allocate()
    # bbb.nis[:, :, com.nhsp] = 1e-6 * bbb.nis[:, :, 0]
    bbb.nis[:, :, com.nhsp] = 1e16
    bbb.nis[:, :, com.nhsp + 1 :] = 1e16
    bbb.ngs[:, :, 1] = 1e16
    initial_short_run()


def scan_density(n_final: float, N_n: int = 10, save_prefix: str = "n_"):
    densities = np.linspace(bbb.ncore[0], n_final, N_n + 1)
    for n in densities[1:]:
        print(
            "*********************** TRYING n = {:.2e} *************************".format(
                n
            )
        )
        bbb.ncore[0] = n
        bbb.icntnunk = 0
        bbb.dtreal = 1e-12
        bbb.exmain()
        rundt()
        if bbb.iterm == 1:
            hdf5_save(save_prefix + "{:.2e}".format(b0) + ".hdf5")
        else:
            break


def switch_to_diff_neuts():
    """Switch from fluid to diffusive neutral model"""

    if bbb.ni.shape[-1] > 2:
        set_h_gas(fluid_neuts=False)
        set_carbon_imps()
        bbb.nis[:, :, 1:-1] = bbb.ni[:, :, 2:]
        bbb.ups[:, :, 1:-1] = bbb.up[:, :, 2:]
        bbb.ngs[:, :, -1] = bbb.ng[:, :, -1]
        bbb.allocate()
        initial_short_run()
    else:
        set_h_gas(fluid_neuts=False)
        bbb.allocate()
        initial_short_run()


def two_zone_diff_coeffs(core: float = 10, sol: float = 1):
    """Create diffusion coefficients differently in core/SOL

    :param core: Value to use in core, defaults to 10
    :param sol: Value to use in SOL, defaults to 1
    """
    coeff_use = np.zeros((com.nx + 2, com.ny + 2))

    if com.nxpt == 1:
        coeff_use[:, :] = sol
        coeff_use[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, : com.iysptrx1[0] + 1] = core
    elif com.nxpt == 2:
        coeff_use[:, :] = sol
        coeff_use[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, : com.iysptrx1[0] + 1] = core

    return coeff_use


def three_zone_diff_coeffs(core: float = 10, sol: float = 1, div: float = 1):
    """Create diffusion coefficients differently in core/SOL/divertor

    :param core: Value to use in core, defaults to 10
    :param sol: Value to use in SOL, defaults to 1
    :param div: Value to use in divertor (i.e. below X-point), defaults to 1
    """
    coeff_use = np.zeros((com.nx + 2, com.ny + 2))

    if com.nxpt == 1:
        coeff_use[:, :] = sol
        coeff_use[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, : com.iysptrx1[0] + 1] = core

        # Set the values below the X-point
        for iy in range(com.ny + 2):
            coeff_use[: com.ixpt1[0] + 1, iy] = div
            coeff_use[com.ixpt2[0] + 1 :, iy] = div

    elif com.nxpt == 2:
        coeff_use[:, :] = sol
        coeff_use[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, : com.iysptrx1[0] + 1] = core

        # Set the values below the X-point
        for iy in range(com.ny + 2):
            coeff_use[: com.ixpt1[0] + 1, iy] = div
            coeff_use[com.ixpt1[1] - 1 :, iy] = div

    return coeff_use


def scan_variable(
    var: str,
    target: float,
    index: int | None = None,
    savedir: str = "variable_scan",
    num_steps: int = 10,
    method: str = "linear",
):
    """Scan a variable from its initial value to the target using rundt

    :param var: Name of variable
    :param target: Target value(s)
    :param index: Index of variable to change if array type
    :param savedir: Directory to save intermediate steps, defaults to "variable_scan"
    :param num_steps: Number of steps to take, defaults to 10
    :param step_method: Step method, options: "linear", "geometric", "inverse", defaults to "linear"
    """

    # Generate the values over which to scan
    initial_val = getattr(bbb, var)
    if index is not None:
        initial_val = initial_val[index]
    if method == "linear":
        vals = np.linspace(initial_val, target, num_steps)
    elif method == "inverse":
        vals = 1 / np.linspace(1 / initial_val, 1 / target, num_steps)

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    # Save the values in a text file for reference
    percs = [100 * i / (num_steps - 1) for i in range(len(vals))]
    np.savetxt(
        os.path.join(savedir, "scan_values.txt"),
        np.array([percs, vals]).T,
        header="Percentage\tValue",
        fmt=["%.2f", "%.15e"],
        delimiter="\t",
    )

    # Save the initial step
    hdf5_save(os.path.join(savedir, "progress_{:.2f}pc.hdf5".format(0.0)))

    # Loop through each value calling rundt() and saving if successful
    for i, val in enumerate(vals[1:]):
        setattr(bbb, var, val)
        bbb.dtreal = 1e-12
        bbb.exmain()
        rundt()
        perc = 100 * (i + 1) / (num_steps - 1)
        if bbb.iterm == 1:
            print("======= PROGRESS = {:.2f}% =======".format(perc))
            hdf5_save(os.path.join(savedir, "progress_{:.2f}pc.hdf5".format(perc)))
        else:
            print("======= RUNDT FAILED WITH PROGRESS = {:.2f}% =======".format(perc))
            break


def generate_variable_scans(
    variables: str | list[str],
    initial: float | np.ndarray | list[np.ndarray],
    final: float | np.ndarray | list[np.ndarray],
    num_steps: int | list[int],
    savepath: str | None = None,
):
    """Generate a dictionary of variables to scan over. This is intended to be used with Slurm array jobs

    :param variables: List of variable names to scan over
    :param initial: List of initial values of variables
    :param final: List of final values of variables
    :param num_steps: List of number of steps over which to scan each variable (from initial to final in linear spacing)
    :param savepath: Savepath of the dictionary
    :return: Dictionary containing variable names and values
    """
    scan_dict = {}
    if isinstance(variables, str):
        variables = [variables]
        initial = [initial]
        final = [final]
        num_steps = [num_steps]
    for i, v in enumerate(variables):
        scan_dict[v] = {}
        for step in range(num_steps[i]):
            scan_dict[v][step] = initial[i] + step / (num_steps[i] - 1) * (
                final[i] - initial[i]
            )

    if savepath is not None:
        if savepath.endswith(".pkl"):
            with open(savepath, "wb") as outfile:
                pickle.dump(
                    scan_dict,
                    outfile,
                )
        elif savepath.endswith(".yml") or savepath.endswith(".yaml"):
            with open(savepath, "w") as outfile:
                yaml.dump(scan_dict, outfile, default_flow_style=False)
        else:
            raise Exception(
                "savepath should end in either '.pkl' to save as a pickle file, or '.yml'/'yaml to save as a yaml file"
            )
    else:
        return scan_dict
