import matplotlib.colors
from uedge import *
import matplotlib.pyplot as plt
import uedge_dcp.post_processing as pp
from numpy import zeros, sum, transpose, mgrid, nan, array, cross, nan_to_num
from scipy.interpolate import griddata, bisplrep
from matplotlib.patches import Polygon
from copy import deepcopy
import matplotlib
from matplotlib.collections import PatchCollection


def plot_q_plates():
    """Plot the total heat flux delivered to each target plate"""
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


def plotvar(
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
):
    """Plot a variable on the UEDGE mesh. Variable must have same dimensions as grid.

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
    p = PatchCollection(patches, norm=norm, cmap="inferno")
    p.set_array(np.array(vals))

    fig, ax = plt.subplots(1)

    ax.add_collection(p)
    ax.autoscale_view()
    plt.colorbar(p, label=label)

    if iso:
        plt.axis("equal")  # regular aspect-ratio

    fig.suptitle(title)
    ax.set_title(subtitle, loc="left")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")

    if grid:
        plt.grid(True)

    if yinv:
        plt.gca().invert_yaxis()

    # if (iso):
    #    plt.axes().set_aspect('equal', 'datalim')
    # else:
    #    plt.axes().set_aspect('auto', 'datalim')

    if show:
        plt.show()
