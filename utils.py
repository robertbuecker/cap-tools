from collections import defaultdict
import csv
from math import radians, cos, floor, log10
from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy.cluster.hierarchy import fcluster
import yaml

def err_str(value, error, errordigits=1, compact=True):
    digits = max(0,-int(floor(log10(error)))) - 1 + errordigits    
    if compact:
        return "{0:.{2}f}({1:.0f})".format(value, error*10**digits, digits)
    else:
        return '{0:.{2}f} ({1:.{2}f})'.format(value, error, digits)

def space_group_lib():
    """Initialize simple space group library mapping the space group 
    number to a dict with information on the `class` (crystal class),
    `lattice` (lattice symbol), `laue_symmetry` (number of the lowest 
    symmetry space group for this lattice), `name` (space group name), 
    and `number` (space group number)."""
    fn = Path(__file__).parent / "spglib.yaml"
    return yaml.load(open(fn, "r"), Loader=yaml.Loader)


def volume(cell):
    """Returns volume for the general case from cell parameters"""
    a, b, c, al, be, ga = cell
    al = radians(al)
    be = radians(be)
    ga = radians(ga)
    vol = a*b*c * \
        ((1+2*cos(al)*cos(be)*cos(ga)-cos(al)**2-cos(be)**2-cos(ga)**2)
         ** .5)
    return vol


def parse_args_for_fns(args, name="XDS.INP", match=None):
    """Parse list of filenames and resolve wildcards
    name:
        Name of the file to locate
    match:
        Match the file list against the provided glob-style pattern.
        If the match is False, the path is removed from the list.
        example:
            match="SMV_reprocessed"
        """
    if not args:
        fns = [Path(".")]
    else:
        fns = [Path(fn) for fn in args]

    new_fns = []
    for fn in fns:
        if fn.is_dir():
            new_fns.extend(list(fn.rglob(f"{name}")))
        else:  
            new_fns.append(fn)
    
    if match:
        new_fns = [fn for fn in new_fns if fn.match(f"{match}/*")]
    
    new_fns = [fn.resolve() for fn in new_fns]

    print(f"{len(new_fns)} files named {name} (subdir: {match}) found.")

    return new_fns


def weighted_average(values, weights=None):
    """Returns weighted mean and standard deviation"""
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean)**2, weights=weights)
    std = np.sqrt(variance)
    return mean, std


def write_cap_csv(fn: str, ds: List[dict]):
    """Writes a structure of results in csv.DictWriter format into CrysAlisPro ED result viewer CSV format"""

    from textwrap import dedent
    with open(fn, 'w', newline='') as fh:
        fh.write(dedent(
            f'''\
            VERSION 1
            HEADER INFO: 
            Number of experiments: {len(ds)}
            Number of columns {len(ds[0])}



            '''
                ))
        writer = csv.DictWriter(fh, fieldnames=ds[0].keys())
        writer.writeheader()
        for experiment in ds:
            writer.writerow(experiment)


def parse_cap_csv(fn: str, use_raw_cell: bool, filter_missing: bool = True) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """Parses a CSV experiment report file from the CrysAlisPro ED result viewer"""

    with open(fn) as fh:
        for _ in range(7):
            _ = fh.readline()
        ds = list(csv.DictReader(fh))
    key = 'Current unit cell' if use_raw_cell else 'Final SG unit cell '
    ds = [d for d in ds if d[key]] if filter_missing else ds # filter experiments with empty cell parameters (not indexed)
    cells = np.array([d[key].split() for d in ds]).astype(float)
    weights = np.array([1 for d in ds])

    return ds, cells, weights


def put_in_order(cells: np.ndarray) -> np.ndarray:
    """Order cell parameters in order to eliminate difference in cell distance because of parameter order"""

    ordered_cells = []
    for i in range(0, len(cells)):
        cell = cells[i, :]
        cell_lengths = np.array(cell[:3])
        cell_angles = np.array(cell[3:])
        cell_array = np.vstack((cell_lengths, cell_angles))
        sortedArr = cell_array[:, np.argsort(cell_array[0, :])]
        sortedArr = sortedArr.ravel()
        ordered_cells.append(sortedArr)
    return np.array(ordered_cells)


def to_sin(cells: np.ndarray) -> np.ndarray:
    """convert all angles in unit cell parameter list to sine
    cells: the cell parameters that are parsed from cells.yaml as np array"""
    angles = cells[:, 3:6]
    angles_sine = np.sin(np.radians(angles))

    cells_sine = cells.copy()
    cells_sine[:, 3:6] = angles_sine
    # convert also the cell angles using arcsin in order to avoid the <> 90 degree ambiguity thingy
    cells[:, 3:6] = np.degrees(np.arcsin(cells_sine[:, 3:6]))

    return cells_sine


def to_radian(cells: np.ndarray) -> np.ndarray:
    """convert all angles in unit cell parameter list to radians
    cells: the cell parameters that are parsed from cells.yaml as np array"""
    cells_radian = cells.copy()
    cells_radian[:, 3:6] = np.radians(cells[:, 3:6])

    return cells_radian


def d_calculator(cell: list) -> tuple:
    """Helper function for `unit_cell_lcv_distance`"""
    a, b, c, alpha, beta, gamma = cell
    d_ab = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(np.radians(180 - gamma)))
    d_ac = np.sqrt(a**2 + c**2 - 2*a*c*np.cos(np.radians(180 - beta)))
    d_bc = np.sqrt(b**2 + c**2 - 2*b*c*np.cos(np.radians(180 - alpha)))
    return d_ab, d_ac, d_bc


def unit_cell_lcv_distance(cell1: list, cell2: list) -> float:
    """Implements Linear Cell Volume from Acta Cryst. (2013). D69, 1617-1632"""
    d_ab1, d_ac1, d_bc1 = d_calculator(cell1)
    d_ab2, d_ac2, d_bc2 = d_calculator(cell2)
    M_ab = abs(d_ab1 - d_ab2)/min(d_ab1, d_ab2)
    M_ac = abs(d_ac1 - d_ac2)/min(d_ac1, d_ac2)
    M_bc = abs(d_bc1 - d_bc2)/min(d_bc1, d_bc2)
    return max(M_ab, M_ac, M_bc)


def volume_difference(cell1: list, cell2: list):
    """Return the absolute difference in volumes between two unit cells"""
    v1 = volume(cell1)
    v2 = volume(cell2)
    return abs(v1-v2)


def get_clusters(z, cells: np.ndarray, distance: float = 0.5) -> defaultdict:
    clusters = fcluster(z, distance, criterion='distance')
    grouped = defaultdict(list)
    for i, c in enumerate(clusters):
        grouped[c].append(i)

    print("-"*40)
    np.set_printoptions(formatter={'float': '{:7.2f}'.format})
    for i in sorted(grouped.keys()):
        cluster = grouped[i]
        clustsize = len(cluster)
        if clustsize == 1:
            del grouped[i]
            continue
        print(f"\nCluster #{i} ({clustsize} items)")
        vols = []
        for j in cluster:
            cell = cells[j]
            vol = volume(cell)
            vols.append(vol)
            print(f"{j+1:5d} {cell}  Vol.: {vol:6.1f}")
        print(" ---")
        print("Mean: {}  Vol.: {:6.1f}".format(np.mean(cells[cluster], axis=0), np.mean(vols)))
        print(" Min: {}  Vol.: {:6.1f}".format(np.min(cells[cluster], axis=0), np.min(vols)))
        print(" Max: {}  Vol.: {:6.1f}".format(np.max(cells[cluster], axis=0), np.max(vols)))

    print("")

    return grouped

