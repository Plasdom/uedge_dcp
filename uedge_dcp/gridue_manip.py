import numpy as np
import netCDF4
from uedge import bbb
import shapely as sh


class Grid:
    """Class for reading a gridue file"""

    def __init__(self, geometry: str, filename: str):
        """Initialise a Grid object with a geometry and filepath.

        :param geometry: Geometry of the grid. This should match the nomenclature used in UEDGE, e.g. "snull", "dnbot", "snowflake75", etc.
        :param filename: Filepath for the gridue file
        """
        self.geometry = geometry
        self.locs = ["CENTER", "SW", "SE", "NW", "NE"]
        self.read_grid(filename)

    def read_grid(self, filename) -> None:
        """Parse a gridue file

        :param filename: Full filepath to gridue file
        """
        # TODO: Could just use call to  uedge.readgrid() here?
        # Read in the lines
        with open(filename) as f:
            lines = f.readlines()
        nlines = len(lines)

        # Find the header
        body_start = 0
        for i in range(nlines):
            if lines[i] == "\n":
                body_start = i + 1
                break
        header_lines = lines[: body_start - 1]

        # Parse the header
        if self.geometry == "dnbot":
            header_data = header_lines[0].strip("\n").split()
            self.nx = int(header_data[0])
            self.ny = int(header_data[1])
            self.ixpt1 = int(header_data[2])
            self.ixp2 = int(header_data[3])
            self.iyseptrx1 = int(header_data[4])
        else:
            header_data = [hl.strip("\n").split() for hl in header_lines]
            self.nx = int(header_data[0][0])
            self.ny = int(header_data[0][1])
            self.iyseparatrix1 = int(header_data[1][0])
            self.iyseparatrix2 = int(header_data[1][1])
            self.ix_plate1 = int(header_data[2][0])
            self.ix_cut1 = int(header_data[2][1])
            self.ix_cut2 = int(header_data[2][3])
            self.ix_plate2 = int(header_data[2][4])
            self.iyseparatrix3 = int(header_data[3][0])
            self.iyseparatrix4 = int(header_data[3][1])
            self.ix_plate3 = int(header_data[4][0])
            self.ix_cut3 = int(header_data[4][1])
            self.ix_cut4 = int(header_data[4][3])
            self.ix_plate4 = int(header_data[4][4])

        # Locate each block of data
        breaks = [body_start]
        for i in range(body_start, nlines):
            if lines[i] == "\n" or "iogridue" in lines[i]:
                breaks.append(i)

        # Read each block of data
        grid_vars = []
        for i in range(8):
            var_lines = lines[breaks[i] : breaks[i + 1]]
            var_data = np.array(
                [
                    float(v.replace("D", "e"))
                    for l in var_lines
                    for v in l.strip("\n").split()
                ]
            )
            var_array = np.zeros((self.nx + 2, self.ny + 2, len(self.locs)))
            i_count = 0
            for iloc in range(len(self.locs)):
                for iy in range(self.ny + 2):
                    for ix in range(self.nx + 2):
                        var_array[ix, iy, iloc] = var_data[i_count]
                        i_count += 1

            grid_vars.append(var_array)

        # Save grid vars
        self.r = grid_vars[0]
        self.z = grid_vars[1]
        self.psi = grid_vars[2]
        self.br = grid_vars[3]
        self.bz = grid_vars[4]
        self.bpol = grid_vars[5]
        self.btor = grid_vars[6]
        self.b = grid_vars[7]

        # Extract cell-centred values
        self.r_c = self.r[:, :, 0]
        self.z_c = self.z[:, :, 0]
        self.psi_c = self.psi[:, :, 0]
        self.br_c = self.br[:, :, 0]
        self.bz_c = self.bz[:, :, 0]
        self.bpol_c = self.bpol[:, :, 0]
        self.btor_c = self.btor[:, :, 0]
        self.b_c = self.b[:, :, 0]


class UESave:
    def __init__(self, filename: str):

        self.varlist = ["te", "ti", "phi", "ni", "ng", "up", "tg"]

        ds = netCDF4.Dataset(filename)
        self.rm = ds["com"]["rm"][:]
        self.zm = ds["com"]["zm"][:]
        self.nx = ds["com"]["nx"][:]
        self.ny = ds["com"]["ny"][:]
        self.vars = {}
        for v in self.varlist:
            self.vars[v] = ds["bbb"][v][:]


from scipy.spatial import cKDTree as KDTree


class Invdisttree:
    """
    This code was taken from Stack Overflow user denis on 2nd May 2025: https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python

    inverse-distance-weighted interpolation using KDTree:
    invdisttree = Invdisttree( X, z )  -- data points, values
    interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
        interpolates z from the 3 points nearest each query point q;
        For example, interpol[ a query point q ]
        finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
            (z1/d1 + z2/d2 + z3/d3)
            / (1/d1 + 1/d2 + 1/d3)
            = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

        q may be one point, or a batch of points.
        eps: approximate nearest, dist <= (1 + eps) * true nearest
        p: use 1 / distance**p
        weights: optional multipliers for 1 / distance**p, of the same shape as q
        stat: accumulate wsum, wn for average weights

    How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
        |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
        I find that runtimes don't increase much at all with nnear -- ymmv.

    p=1, p=2 ?
        p=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2,
        so p=2 is inverse-area weighting. For example,
            (z1/area1 + z2/area2 + z3/area3)
            / (1/area1 + 1/area2 + 1/area3)
            = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
        Similarly, in 3d, p=3 is inverse-volume weighting.

    Scaling:
        if different X coordinates measure different things, Euclidean distance
        can be way off.  For example, if X0 is in the range 0 to 1
        but X1 0 to 1000, the X1 distances will swamp X0;
        rescale the data, i.e. make X0.std() ~= X1.std() .

    A nice property of IDW is that it's scale-free around query points:
    if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
    the IDW average
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
    is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
    In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
    is exceedingly sensitive to distance and to h.

    """

    # anykernel( dj / av dj ) is also scale-free
    # error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__(self, X, z, leafsize=10, stat=0):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree(X, leafsize=leafsize)  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None

    def __call__(self, q, nnear=6, eps=0, p=1, weights=None):
        # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query(q, k=nnear, eps=eps)
        interpol = np.zeros((len(self.distances),) + np.shape(self.z[0]))
        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot(w, self.z[ix])
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1 else interpol[0]


def interpolate_var_kde(
    g1: Grid,
    g2: Grid,
    uevar: np.ndarray,
    leafsize: int = 10,
    nnear: int = 8,
    eps: float = 0.1,
    p: int = 2,
    squeeze: bool = True,
) -> np.ndarray:
    """Interpolate a variable from an old grid and savefile to a new grid

    :param old_grid: Old grid
    :param new_grid: New grid
    :param uevar: UEDGE variable name
    :param leafsize: Leafsize for KDTree interpolation, defaults to 10
    :param nnear: Nearest neighbours for interpolation, defaults to 8
    :param eps: Eps for interpolation, defaults to 0.1
    :param p: p for interpolation, defaults to 2
    :return: Interpolated variable
    """

    # Carry out interpolation
    known = np.array([g1.r_c.ravel(), g1.z_c.ravel()]).T
    ask = np.array([g2.r_c.ravel(), g2.z_c.ravel()]).T
    z = uevar
    if len(z.shape) == 2:
        nsp = 1
        z = [z.ravel()]
    elif len(z.shape) == 3:
        nsp = z.shape[-1]
        z = []
        for isp in range(nsp):
            z.append(uevar[:, :, isp].ravel())

    z_new = np.zeros((g2.nx + 2, g2.ny + 2, nsp))
    for i in range(nsp):
        invdisttree = Invdisttree(known, z[i], leafsize=leafsize, stat=1)
        interpol = invdisttree(ask, nnear=nnear, eps=eps, p=p)

        # TODO: Carry out the above for all variables (will need to read in ngsp, nisp into UESave class, etc)
        # TODO: Reshape interpolated variables to the shape of the new grid
        # TODO: We can now either save the interpolated variables into a new hdf5 file, or just call this function in a uedge run script and set nis, tes, etc accordingly (latter is probably easier as don't have to faff around with hdf5 formatting)

        # Reshape
        interpol = interpol.reshape(g2.r_c.shape)
        z_new[:, :, i] = interpol

    if nsp == 1 and squeeze:
        z_new = z_new.squeeze()

    return z_new


def get_grid_overlaps(g1: Grid, g2: Grid) -> list:
    """Get the overlaps between the cells of two UEDGE grids. Output is a 2D list (with same dims as grid g2), where each entry
    is a list of tuples for each overlapping cell. The values of each tuple are x index of overlapping g1 cell, y index of overlapping g1 cell, overlap area

    :param g1: Old grid
    :param g2: New grid
    :return: overlaps information
    """
    # Create polygons for the cells of the old and new grid
    old_cells = [[None for i in range(g1.ny + 2)] for j in range(g1.nx + 2)]
    for ix in range(g1.nx + 2):
        for iy in range(g1.ny + 2):
            cell_coords = np.array(
                [g1.r[ix, iy, [1, 2, 4, 3, 1]], g1.z[ix, iy, [1, 2, 4, 3, 1]]]
            ).T
            old_cells[ix][iy] = sh.Polygon(cell_coords)
    new_cells = [[None for i in range(g2.ny + 2)] for j in range(g2.nx + 2)]
    for ix in range(g2.nx + 2):
        for iy in range(g2.ny + 2):
            cell_coords = np.array(
                [g2.r[ix, iy, [1, 2, 4, 3, 1]], g2.z[ix, iy, [1, 2, 4, 3, 1]]]
            ).T
            new_cells[ix][iy] = sh.Polygon(cell_coords)

    # Find overlaps
    overlaps = [[None for i in range(g2.ny + 2)] for j in range(g2.nx + 2)]
    for ix2 in range(g2.nx + 2):
        for iy2 in range(g2.ny + 2):
            new_cell = new_cells[ix2][iy2]
            overlaps[ix2][iy2] = []
            for ix1 in range(g1.nx + 2):
                for iy1 in range(g1.ny + 2):
                    old_cell = old_cells[ix1][iy1]
                    overlap = sh.intersection(new_cell, old_cell)
                    if overlap.area > 0:
                        overlaps[ix2][iy2].append((ix1, iy1, overlap.area))

    return overlaps


def interpolate_var_overlaps(
    g1: Grid,
    g2: Grid,
    uevar: np.ndarray,
    overlaps: list = None,
    squeeze: bool = True,
    default_val: float = 0,
) -> np.ndarray:
    """Interpolate a variable from an old grid and savefile to a new grid

    :param old_grid: Old grid
    :param new_grid: New grid
    :param uevar: UEDGE variable array
    :return: Interpolated variable
    """

    if overlaps is None:
        # Create polygons for the cells of the old and new grid
        old_cells = [[None for i in range(g1.ny + 2)] for j in range(g1.nx + 2)]
        for ix in range(g1.nx + 2):
            for iy in range(g1.ny + 2):
                cell_coords = np.array(
                    [g1.r[ix, iy, [1, 2, 4, 3, 1]], g1.z[ix, iy, [1, 2, 4, 3, 1]]]
                ).T
                old_cells[ix][iy] = sh.Polygon(cell_coords)
        new_cells = [[None for i in range(g2.ny + 2)] for j in range(g2.nx + 2)]
        for ix in range(g2.nx + 2):
            for iy in range(g2.ny + 2):
                cell_coords = np.array(
                    [g2.r[ix, iy, [1, 2, 4, 3, 1]], g2.z[ix, iy, [1, 2, 4, 3, 1]]]
                ).T
                new_cells[ix][iy] = sh.Polygon(cell_coords)

        # Find overlaps
        overlaps = [[None for i in range(g2.ny + 2)] for j in range(g2.nx + 2)]
        for ix2 in range(g2.nx + 2):
            for iy2 in range(g2.ny + 2):
                new_cell = new_cells[ix2][iy2]
                overlaps[ix2][iy2] = []
                for ix1 in range(g1.nx + 2):
                    for iy1 in range(g1.ny + 2):
                        old_cell = old_cells[ix1][iy1]
                        overlap = sh.intersection(new_cell, old_cell)
                        if overlap.area > 0:
                            overlaps[ix2][iy2].append([ix1, iy1, overlap.area])

    # Carry out overlap based interpolation
    if len(uevar.shape) == 2:
        nsp = 1
    elif len(uevar.shape) == 3:
        nsp = uevar.shape[-1]
    z_new = np.zeros([g2.nx + 2, g2.ny + 2, nsp])
    for ix2 in range(g2.nx + 2):
        for iy2 in range(g2.ny + 2):
            ols = overlaps[ix2][iy2]
            for ol in ols:
                ix1 = ol[0]
                iy1 = ol[1]
                z_new[ix2, iy2] += uevar[ix1][iy1] * ol[-1]
            total_area = np.sum(ol[-1] for ol in ols)
            if total_area > 0:
                z_new[ix2, iy2] = z_new[ix2, iy2] / total_area
            else:
                # No overlap found between current cell and old grid - revert to default value
                z_new[ix2, iy2] = default_val

    if nsp == 1 and squeeze:
        z_new = z_new.squeeze()

    return z_new


def interpolate_save(
    old_grid: str,
    new_grid: str,
    old_save: str,
    geometry: str = "dnbot",
    method: str = "overlaps",
) -> None:
    """Update the initial variable values for a current UEDGE session by interpolating from an old grid+save to the new grid

    :param old_grid: Filepath to old grid
    :param new_grid: Filepath to new grid
    :param old_save: Filepath to old HDF5 save
    :param geometry: Grid geometry, defaults to "dnbot"
    :param method: Inteprolation method ["KDE", "overlaps"], defaults to "overlaps"
    """
    g1 = Grid(
        geometry,
        old_grid,
    )
    g2 = Grid(
        geometry,
        new_grid,
    )

    # Read old save file
    oldsave = UESave(old_save)

    if method == "KDE":
        leafsize = 10
        nnear = 8
        eps = 0.1
        p = 2
        bbb.tes = interpolate_var_kde(
            g1, g2, oldsave.vars["te"], leafsize, nnear, eps, p
        )
        bbb.tis = interpolate_var_kde(
            g1, g2, oldsave.vars["ti"], leafsize, nnear, eps, p
        )
        bbb.nis = interpolate_var_kde(
            g1, g2, oldsave.vars["ni"], leafsize, nnear, eps, p
        )
        bbb.ngs = interpolate_var_kde(
            g1, g2, oldsave.vars["ng"], leafsize, nnear, eps, p, squeeze=False
        )
        bbb.phis = interpolate_var_kde(
            g1, g2, oldsave.vars["phi"], leafsize, nnear, eps, p
        )
        bbb.ups = interpolate_var_kde(
            g1, g2, oldsave.vars["up"], leafsize, nnear, eps, p
        )
    elif method == "overlaps":
        overlaps = get_grid_overlaps(g1, g2)
        bbb.tes = interpolate_var_overlaps(
            g1, g2, oldsave.vars["te"], overlaps, default_val=10 * bbb.ev
        )
        bbb.tis = interpolate_var_overlaps(
            g1, g2, oldsave.vars["ti"], overlaps, default_val=10 * bbb.ev
        )
        bbb.nis = interpolate_var_overlaps(
            g1, g2, oldsave.vars["ni"], overlaps, default_val=1e16
        )
        bbb.ngs = interpolate_var_overlaps(
            g1, g2, oldsave.vars["ng"], overlaps, squeeze=False, default_val=1e16
        )
        bbb.phis = interpolate_var_overlaps(
            g1, g2, oldsave.vars["phi"], overlaps, default_val=0.0
        )
        bbb.ups = interpolate_var_overlaps(
            g1, g2, oldsave.vars["up"], overlaps, default_val=0.0
        )
