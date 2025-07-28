import matplotlib.colors
from uedge import *
import matplotlib.pyplot as plt
import uedge_dcp.post_processing as pp
from uedge_dcp.gridue_manip import Grid
from numpy import zeros, sum, transpose, mgrid, nan, array, cross, nan_to_num
from scipy.interpolate import griddata, bisplrep
from matplotlib.patches import Polygon
from copy import deepcopy
import matplotlib
from matplotlib.collections import PatchCollection
from numpy import zeros, sum, transpose, mgrid, nan, array, cross, nan_to_num
from scipy.interpolate import griddata, bisplrep
from matplotlib.patches import Polygon
from copy import deepcopy
from matplotlib.widgets import Slider


def plot_q_plates():
    """Plot the total heat flux delivered to each target plate (snowflake geometry is assumed)"""
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

    P1, P2, P3, P4 = pp.get_Q_target_proportions()
    Ptot = P1 + P2 + P3 + P4

    print("Power delivered to each target plate: ")
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

    fig, ax = plt.subplots(4, 1)

    (c,) = ax[0].plot(r1, q1 / 1e6, label="SP1")
    ax[1].plot(r2, q2 / 1e6, label="SP2")
    ax[2].plot(r3, q3 / 1e6, label="SP3")
    ax[3].plot(r4, q4 / 1e6, label="SP4")

    [ax[i].grid() for i in range(4)]
    [ax[i].legend(loc="upper right") for i in range(4)]
    ax[1].set_ylabel("Heat flux [MWm$^{-2}$]")
    ax[-1].set_xlabel("Distance along target plate [m]")
    fig.tight_layout()


def comparemesh(
    gridfile1: str,
    gridfile2: str,
    geom1: str = "snowflake75",
    geom2: str = "snowflake75",
):
    g1 = Grid(geom1, gridfile1)
    g2 = Grid(geom2, gridfile2)

    fig, ax = plt.subplots(1)
    ax.set_aspect("equal")

    for iy in np.arange(0, g1.ny + 2):
        for ix in np.arange(0, g1.nx + 2):
            ax.plot(
                g1.r[ix, iy, [1, 2, 4, 3, 1]],
                g1.z[ix, iy, [1, 2, 4, 3, 1]],
                color="black",
                linewidth=0.5,
            )

    for iy in np.arange(0, g2.ny + 2):
        for ix in np.arange(0, g2.nx + 2):
            ax.plot(
                g2.r[ix, iy, [1, 2, 4, 3, 1]],
                g2.z[ix, iy, [1, 2, 4, 3, 1]],
                color="red",
                linestyle="--",
                linewidth=0.5,
            )

    ax.plot([], [], color="black", label="Grid 1")
    ax.plot([], [], color="red", linestyle="--", label="Grid 2")
    ax.legend(loc="upper right")


def plotmesh(
    gridue_file=None,
    iso=True,
    zshift=0.0,
    xlim=None,
    ylim=None,
    yinv=False,
    title="UEDGE grid",
    subtitle=None,
    show=True,
):

    if gridue_file is None:
        rm = com.rm
        zm = com.zm
        nx = com.nx
        ny = com.ny
    else:
        grid = Grid(geometry="NA", filename=gridue_file)
        rm = grid.r
        zm = grid.z
        nx = grid.nx
        ny = grid.ny

    fig, ax = plt.subplots(1)

    if iso:
        ax.set_aspect("equal", "datalim")
    else:
        ax.set_aspect("auto", "datalim")

    # plt.plot([np.min(com.rm),np.max(com.rm)], [np.min(com.zm),np.max(com.zm)])

    for iy in np.arange(0, ny + 2):
        for ix in np.arange(0, nx + 2):
            ax.plot(
                rm[ix, iy, [1, 2, 4, 3, 1]],
                zm[ix, iy, [1, 2, 4, 3, 1]] + zshift,
                color="black",
                linewidth=0.5,
            )

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    # fig.suptitle('UEDGE grid')
    # plt.title('UEDGE grid')
    # ax.set_subtitle(title)
    ax.set_title(title, loc="left")
    ax.grid(False)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if yinv:
        ax.invert_yaxis()

    if show:
        plt.show()


def plotvar(
    var: np.ndarray,
    rm: np.ndarray = None,
    zm: np.ndarray = None,
    iso: bool = True,
    mesh: bool = False,
    grid: bool = False,
    label: str = None,
    vmin: float = None,
    vmax: float = None,
    yinv: bool = False,
    title: str = "UEDGE data",
    subtitle: str = None,
    show: bool = True,
    logscale: bool = False,
    xlim: tuple = (None, None),
    ylim: tuple = (None, None),
    cmap: str = "inferno",
):
    """Plot a variable on the UEDGE mesh. Variable must have same dimensions as grid.

    :param var: A numpy array to plot
    :param rm: R coordinates (if None, get from uedge.com)
    :param zm: Z coordinates (if None, get from uedge.com)
    :param iso: Plot on axes with equal aspect ratio, defaults to True
    :param mesh: Show the UEDGE mesh lines
    :param grid: Show the grid cells, defaults to False
    :param label: Colour bar label, defaults to None
    :param vmin: vmin for colour bar, defaults to None
    :param vmax: vmax for colour bar, defaults to None
    :param yinv: Invert y axis, defaults to False
    :param title: Plot title, defaults to "UEDGE data"
    :param subtitle: Plot subtitle, defaults to None
    :param show: Call plt.show(), defaults to True
    """

    patches = []

    if rm is None:
        rm = com.rm
        zm = com.zm
        nx = com.nx
        ny = com.ny
    else:
        nx = rm.shape[0] - 2
        ny = rm.shape[1] - 2

    for iy in np.arange(0, ny + 2):
        for ix in np.arange(0, nx + 2):
            rcol = rm[ix, iy, [1, 2, 4, 3]]
            zcol = zm[ix, iy, [1, 2, 4, 3]]
            rcol.shape = (4, 1)
            zcol.shape = (4, 1)
            polygon = Polygon(np.column_stack((rcol, zcol)))
            patches.append(polygon)

    # -is there a better way to cast input data into 2D array?
    vals = np.zeros((nx + 2) * (ny + 2))

    for iy in np.arange(0, ny + 2):
        for ix in np.arange(0, nx + 2):
            k = ix + (nx + 2) * iy
            vals[k] = var[ix, iy]

    # Set vmin and vmax disregarding guard cells
    if not vmax:
        vmax = np.max(var)
    if not vmin:
        vmin = np.min(var)

    if logscale:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    ###p = PatchCollection(patches, cmap=cmap, norm=norm)
    if mesh:
        lw = 0.1
    else:
        lw = 1e-6
    p = PatchCollection(
        patches, norm=norm, cmap=cmap, edgecolors="black", linewidths=lw
    )

    p.set_array(np.array(vals))

    fig, ax = plt.subplots(1)

    ax.add_collection(p)
    ax.autoscale_view()
    plt.colorbar(p, label=label)

    if iso:
        plt.axis("equal")  # regular aspect-ratio

    fig.suptitle(title)
    ax.set_title(subtitle, loc="left")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if grid:
        ax.grid()

    if yinv:
        ax.invert_yaxis()

    # if (iso):
    #    plt.axes().set_aspect('equal', 'datalim')
    # else:
    #    plt.axes().set_aspect('auto', 'datalim')

    # if show:
    #     plt.show()


def animatevar(
    var: np.ndarray,
    iso: bool = True,
    grid: bool = False,
    label: str = None,
    vmin: float = None,
    vmax: float = None,
    yinv: bool = False,
    title: str = "UEDGE data",
    subtitle: str = None,
    show: bool = True,
    logscale: bool = False,
    xlim: tuple = (None, None),
    ylim: tuple = (None, None),
):
    """Animate a variable on the UEDGE mesh. Variable must have same dimensions as grid, plus a timestep dimension (assumed to be the last dimension in the array).

    :param var: A numpy array to plot
    :param iso: Plot on axes with equal aspect ratio, defaults to True
    :param grid: Show the grid cells, defaults to False
    :param label: Colour bar label, defaults to None
    :param vmin: vmin for colour bar, defaults to None
    :param vmax: vmax for colour bar, defaults to None
    :param yinv: Invert y axis, defaults to False
    :param title: Plot title, defaults to "UEDGE data"
    :param subtitle: Plot subtitle, defaults to None
    :param show: Call plt.show(), defaults to True
    """

    patches = []

    for iy in np.arange(0, com.ny + 2):
        for ix in np.arange(0, com.nx + 2):
            rcol = com.rm[ix, iy, [1, 2, 4, 3]]
            zcol = com.zm[ix, iy, [1, 2, 4, 3]]
            rcol.shape = (4, 1)
            zcol.shape = (4, 1)
            polygon = Polygon(np.column_stack((rcol, zcol)))
            patches.append(polygon)

    # -is there a better way to cast input data into 2D array?
    vals = np.zeros((com.nx + 2) * (com.ny + 2))

    for iy in np.arange(0, com.ny + 2):
        for ix in np.arange(0, com.nx + 2):
            k = ix + (com.nx + 2) * iy
            vals[k] = var[ix, iy, 0]

    # Set vmin and vmax disregarding guard cells
    if not vmax:
        vmax = np.max(var)
    if not vmin:
        vmin = np.min(var)

    if logscale:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    ###p = PatchCollection(patches, cmap=cmap, norm=norm)
    p = [PatchCollection(patches, norm=norm, cmap="inferno")]
    p[0].set_array(np.array(vals))

    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0.15, bottom=0.2, right=0.85, top=0.92)
    ax.add_collection(p[0])
    ax.autoscale_view()
    plt.colorbar(p[0], label=label)

    if iso:
        plt.axis("equal")  # regular aspect-ratio

    fig.suptitle(title)
    ax.set_title(subtitle, loc="left")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if grid:
        ax.grid()

    if yinv:
        ax.invert_yaxis()

    axsurf1 = fig.add_axes([0.15, 0.05, 0.6, 0.03])
    surf1_slider = Slider(
        ax=axsurf1,
        label=r"Timestep",
        valmin=0,
        valmax=var.shape[-1] - 1,
        valinit=0,
        valstep=1,
    )

    def update_surf1(val):
        p[0].remove()
        for iy in np.arange(0, com.ny + 2):
            for ix in np.arange(0, com.nx + 2):
                k = ix + (com.nx + 2) * iy
                vals[k] = var[ix, iy, int(val)]
        p[0].set_array(np.array(vals))
        ax.add_collection(p[0])

        return p

    surf1_slider.on_changed(update_surf1)

    return surf1_slider


def getomit(var):
    """Helper function to handled partial grids w/ omits"""
    nxomit = 0
    nyomit = 0
    if isinstance(var, str):
        var = var
    if nyomit > 0:
        var = var[:, :-nyomit]
    return var[nxomit:]


def streamplotvar(
    pol: np.ndarray,
    rad: np.ndarray,
    background_var: np.ndarray = None,
    background_var_label: str = None,
    resolution=(500j, 800j),
    linewidth="magnitude",
    broken_streamlines=True,
    color="red",
    maxlength=0.4,
    mask=True,
    density=2,
    linewidth_mult=1,
    xlim=(None, None),
    ylim=(None, None),
    logscale=False,
    title: str = "",
    **kwargs,
):
    """Plot streamlines of a vector variable with poloidal and radial components (pol, rad). Based on the function streamline() in UETools (https://github.com/LLNL/UETOOLS/blob/aaa823222ecc8ae76647aa8bf5299cd9804b61b1/src/uetools/UePlot/Plot.py)

    :param pol: Poloidal component of variable
    :param rad: Radial component of variable
    :param resolution: Resolution, defaults to (500j, 800j)
    :param linewidth: Streamline width (can be a float, "magnitude" or "absolute"), defaults to "magnitude"
    :param broken_streamlines: Whether to plot broken streamlines or not, defaults to False
    :param color: Colour of streamlines, defaults to "red"
    :param maxlength: Max length of streamlines, defaults to 0.4
    :param mask: Mask, defaults to True
    :param density: Density of streamlines, defaults to 2
    :param linewidth_mult: Linewidth multiplier (useful if using linewidth="magnitude"), defaults to 1
    :param xlim: xlim, defaults to (None, None)
    :param ylim: ylim, defaults to (None, None)
    """

    rm = com.rm
    zm = com.zm
    nx = com.nx
    ny = com.ny

    nodes = zeros((nx + 2, ny + 2, 5, 2))
    nodes[:, :, :, 0] = getomit(rm)
    nodes[:, :, :, 1] = getomit(zm)
    nodes = transpose(nodes, (2, 3, 0, 1))

    # TODO: rather than align poloidal/radial in direction of cell,
    # evaluate face normal locally at cell face?
    # Find midpoints of y-faces
    symid = zeros((2, 2, nx + 2, ny + 2))
    symid[0] = (nodes[2] + nodes[1]) / 2  # Lower face center
    symid[1] = (nodes[4] + nodes[3]) / 2  # Upper face center
    # Find midpoints of x-faces
    sxmid = zeros((2, 2, nx + 2, ny + 2))
    sxmid[0] = (nodes[3] + nodes[1]) / 2  # Left face center
    sxmid[1] = (nodes[4] + nodes[2]) / 2  # Right face center

    # Find vectors of east faces
    eastface = zeros((3, nx + 2, ny + 2))
    eastface[:-1] = nodes[4] - nodes[2]
    # Find vectors of north faces
    northface = zeros((3, nx + 2, ny + 2))
    northface[:-1] = nodes[4] - nodes[3]
    # Find normals to faces
    toroidal = zeros((3, nx + 2, ny + 2))
    toroidal[-1] = 1
    eastnormal = cross(eastface, toroidal, axis=0)
    northnormal = cross(toroidal, northface, axis=0)

    northnormaln = zeros((2, nx + 2, ny + 2))
    for i in range(2):
        northnormaln[i] = northnormal[i] / (sum(northnormal**2, axis=0) ** 0.5 + 1e-20)
    eastnormaln = zeros((2, nx + 2, ny + 2))
    for i in range(2):
        eastnormaln[i] = eastnormal[i] / (sum(eastnormal**2, axis=0) ** 0.5 + 1e-20)

    # Create polygons for masking
    outerx = []
    outerx = outerx + list(rm[::-1][-com.ixpt1[0] :, 0, 2])
    outerx = outerx + list(rm[0, :, 1])
    outerx = outerx + list(rm[:, -1, 3])
    outerx = outerx + list(rm[:, ::-1][-1, :, 4])
    outerx = outerx + list(rm[::-1][: nx - com.ixpt2[0], 0, 1])
    outery = []
    outery = outery + list(zm[::-1][-com.ixpt1[0] :, 0, 2])
    outery = outery + list(zm[0, :, 1])
    outery = outery + list(zm[:, -1, 3])
    outery = outery + list(zm[:, ::-1][-1, :, 4])
    outery = outery + list(zm[::-1][: nx - com.ixpt2[0], 0, 1])

    innerx = rm[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, 0, 1]
    innery = zm[com.ixpt1[0] + 1 : com.ixpt2[0] + 1, 0, 1]

    outer = Polygon(
        array([outerx, outery]).transpose(),
        closed=True,
        facecolor="white",
        edgecolor="none",
    )
    inner = Polygon(
        array([innerx, innery]).transpose(),
        closed=True,
        facecolor="white",
        edgecolor="none",
    )
    x = pol * eastnormaln[0] + rad * northnormaln[0]
    y = pol * eastnormaln[1] + rad * northnormaln[1]

    gx, gy = mgrid[
        rm.min() : rm.max() : resolution[0], zm.min() : zm.max() : resolution[1]
    ]

    xinterp = griddata(
        (sxmid[1, 0, 1:-1, 1:-1].ravel(), sxmid[1, 1, 1:-1, 1:-1].ravel()),
        x[1:-1, 1:-1].ravel(),
        (gx, gy),
    )
    yinterp = griddata(
        (symid[1, 0, 1:-1, 1:-1].ravel(), symid[1, 1, 1:-1, 1:-1].ravel()),
        y[1:-1, 1:-1].ravel(),
        (gx, gy),
    )

    if mask is True:
        for i in range(gx.shape[0]):
            for j in range(gx.shape[1]):
                p = (gx[i, j], gy[i, j])
                if (inner.contains_point(p)) or (not outer.contains_point(p)):
                    xinterp[i, j] = nan
                    yinterp[i, j] = nan

    if linewidth == "magnitude":
        linewidth = (xinterp**2 + yinterp**2) ** 0.5
        linewidth = linewidth.transpose()
        maxwidth = nan_to_num(deepcopy(linewidth)).max()
        linewidth /= maxwidth
    elif linewidth == "absolute":
        linewidth = (xinterp**2 + yinterp**2) ** 0.5
        linewidth = linewidth.transpose()

    linewidth *= linewidth_mult

    fig, ax = plt.subplots(1)
    ax.set_aspect("equal")

    for iy in np.arange(0, com.ny + 2):
        for ix in np.arange(0, com.nx + 2):
            ax.plot(
                rm[ix, iy, [1, 2, 4, 3, 1]],
                zm[ix, iy, [1, 2, 4, 3, 1]],
                color="black",
                linewidth=0.05,
                alpha=0.5,
            )

    if background_var is not None:
        patches = []
        for iy in np.arange(0, ny + 2):
            for ix in np.arange(0, nx + 2):
                rcol = rm[ix, iy, [1, 2, 4, 3]]
                zcol = zm[ix, iy, [1, 2, 4, 3]]
                rcol.shape = (4, 1)
                zcol.shape = (4, 1)
                polygon = Polygon(np.column_stack((rcol, zcol)))
                patches.append(polygon)

        vals = np.zeros((nx + 2) * (ny + 2))

        for iy in np.arange(0, ny + 2):
            for ix in np.arange(0, nx + 2):
                k = ix + (nx + 2) * iy
                vals[k] = background_var[ix, iy]

        # Set vmin and vmax disregarding guard cells
        vmax = np.max(background_var)
        vmin = np.min(background_var)

        if logscale:
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        ###p = PatchCollection(patches, cmap=cmap, norm=norm)
        p = PatchCollection(patches, norm=norm, cmap="inferno")
        p.set_array(np.array(vals))

        ax.add_collection(p)
        ax.autoscale_view()
        plt.colorbar(p, label=background_var_label)

    # ax.streamplot(
    #         gx.transpose(),
    #         gy.transpose(),
    #         xinterp.transpose(),
    #         yinterp.transpose(),
    #         linewidth=0.2,
    #         broken_streamlines=False,
    #         color=color,
    #         density=density,
    #         arrowsize=0,
    #         zorder=9998,
    #     )
    ax.streamplot(
        gx.transpose(),
        gy.transpose(),
        xinterp.transpose(),
        yinterp.transpose(),
        linewidth=linewidth,
        broken_streamlines=broken_streamlines,
        color=color,
        maxlength=maxlength,
        density=density,
        zorder=9999,
        **kwargs,
    )

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if title != "":
        ax.set_title(title)


def plotcell(i: list):
    """Highlight a specific cell

    :param i: A tuple containing cell indices (ix,iy), or a list of such tuples (e.g. [(ix1,iy1),(ix2,iy2),...])
    """
    if isinstance(i[0], int):
        i = [i]

    ch = np.zeros((com.nx + 2, com.ny + 2))

    for cell_idx in i:
        ch[cell_idx[0], cell_idx[1]] = 1.0 + np.random.random() * 0.1

    plotvar(ch)


def plotpslice(iy: int):
    """Highlight a specific poloidal slice

    :param iy: Radial index
    """

    ch = np.zeros((com.nx + 2, com.ny + 2))

    ch[:, iy] = 1

    plotvar(ch)


def plotrslice(ix: list):
    """Highlight a specific radial slice

    :param iy: Poloidal index
    """

    ch = np.zeros((com.nx + 2, com.ny + 2))

    ch[ix, :] = 1

    plotvar(ch)


def plotrprof(
    var: np.ndarray,
    ix: int = -1,
    use_psin: bool = False,
    xlim=None,
    ylim=None,
    xlog=False,
    ylog=False,
    show=True,
):
    """Plot the radial profile of a given variable at a given poloidal cell

    :param var: Variable
    :param ix: _description_, defaults to -1
    :param use_psin: _description_, defaults to False
    :param xlim: _description_, defaults to None
    :param ylim: _description_, defaults to None
    :param xlog: _description_, defaults to False
    :param ylog: _description_, defaults to False
    :param show: _description_, defaults to True
    """

    fig, ax = plt.subplots(1)

    if ix < 0:
        ix0 = bbb.ixmp
    else:
        ix0 = ix

    # if use_psin:
    #     psin = (com.psi - com.simagx) / (com.sibdry - com.simagx)
    #     xcoord = psin[ix0, :, 0]
    #     xlabel = "Psi_norm"
    # else:
    #     xcoord = com.rm[ix0, :, 0] - com.rm[ix0, com.iysptrx, 0]
    #     # xcoord = com.rm[bbb.ixmp, :, 0] - com.rm[bbb.ixmp, com.iysptrx, 0]
    #     xlabel = "rho=R-Rsep [m]"

    ax.plot(var[ix0, :], marker="x")

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    if ylog:
        ax.set_yscale("log")

    if xlog:
        ax.set_xscale("log")

    ax.set_xlabel("iy")
    ax.grid(True)

    if show:
        plt.show()
