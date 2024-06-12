from collections import defaultdict
import csv
from math import radians, cos, floor, log10
from pathlib import Path
from typing import *
import numpy as np
from scipy.cluster.hierarchy import fcluster
import yaml
import tkinter as tk
import tkinter.ttk as ttk

class DisableMixin(object):
    # to disable TreeView in Tkinter
    # from https://stackoverflow.com/questions/52181307/python-tkinter-ttk-how-to-disable-treeview

    def state(self,statespec=None):
        if statespec:
            e = super().state(statespec)
            if 'disabled' in e:
                self.bindtags(self.tags)
            elif '!disabled' in e:
                self.tags = self.bindtags()
                self.bindtags([None])
            return e
        else:
            return super().state()

    def disable(self):
        self.state(('disabled',))

    def enable(self):
        self.state(('!disabled',))

    def is_disabled(self):
        return 'disabled' in self.state()

    def is_enabled(self):
        return not self.is_disabled()
    
class myTreeView(DisableMixin, ttk.Treeview): pass

def err_str(value, error, errordigits=1, compact=True):
    digits = max(0,-int(floor(log10(error)))) - 1 + errordigits    
    if compact:
        return "{0:.{2}f}({1:.0f})".format(value, error*10**digits, digits)
    else:
        return '{0:.{2}f} ({1:.{2}f})'.format(value, error, digits)

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
        
    for d in ds:
        # apply fix for older CAP version bug
        if 'Final SG unit cell ' in d:
            d['Final SG unit cell'] = d['Final SG unit cell ']
            del d['Final SG unit cell ']
        
    key = 'Current unit cell' if use_raw_cell else 'Final SG unit cell'
    ds = [d for d in ds if d[key]] if filter_missing else ds # filter experiments with empty cell parameters (not indexed)
    cells = np.array([d[key].split() for d in ds]).astype(float)
    weights = np.array([1 for d in ds])

    return ds, cells, weights


def order_uc_pars(cells: np.ndarray) -> np.ndarray:
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


def unit_cell_lcv_distance(cell1: list, cell2: list, absolute: bool = False) -> float:
    """Implements Linear Cell Volume from Acta Cryst. (2013). D69, 1617-1632"""
    
    def d_calculator(cell: list) -> tuple:
        """Helper function for `unit_cell_lcv_distance`"""
        a, b, c, alpha, beta, gamma = cell
        d_ab = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(np.radians(180 - gamma)))
        d_ac = np.sqrt(a**2 + c**2 - 2*a*c*np.cos(np.radians(180 - beta)))
        d_bc = np.sqrt(b**2 + c**2 - 2*b*c*np.cos(np.radians(180 - alpha)))
        return d_ab, d_ac, d_bc

    d_ab1, d_ac1, d_bc1 = d_calculator(cell1)
    d_ab2, d_ac2, d_bc2 = d_calculator(cell2)
    M_ab = abs(d_ab1 - d_ab2)/(1 if absolute else min(d_ab1, d_ab2))
    M_ac = abs(d_ac1 - d_ac2)/(1 if absolute else min(d_ac1, d_ac2))
    M_bc = abs(d_bc1 - d_bc2)/(1 if absolute else min(d_bc1, d_bc2))
    return max(M_ab, M_ac, M_bc)


def volume_difference(cell1: list, cell2: list):
    """Return the absolute difference in volumes between two unit cells"""
    v1 = volume(cell1)
    v2 = volume(cell2)
    return abs(v1-v2)


def get_clusters(z: np.ndarray, cells: np.ndarray, distance: float = 0.5) -> DefaultDict[int, List[int]]:
    """Generates clusters from linkage matrix

    Args:
        z (np.ndarray): linkage matrix
        cells (np.ndarray): list of all unit cells
        distance (float, optional): cutoff distance for clustering. Defaults to 0.5.

    Returns:
        Dictionary of clusters with values corresponding to a list of dataset indices
    """
    clusters = fcluster(z, distance, criterion='distance')
    grouped = defaultdict(list)
    for i, c in enumerate(clusters):
        grouped[c].append(i)

    # print cluster parameters
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


def build_merge_tree(z: np.ndarray, distance: float, names: Optional[List[str]] = None) -> Tuple[Tuple[Union[Tuple, str], Union[Tuple, str]], List[int]]:
    """Builds a dendrogram-like "merging tree" from a linkage matrix and a distance cutoff containing
    nested tuples of dataset names. Returns list of tree nodes (containing recursive 2-tuples) and a corresponding list of cluster IDs

    Args:
        z (np.ndarray): linkage matrix
        distance (float): list of all unit cells
        names (List[str], optional): _description_. Defaults to None.

    Returns:
        Tuple[Tuple[Union[Tuple, str], Union[Tuple, str]], List[int]]: _description_
    """

    merge_list = []
    cluster_list = []
    cluster_id = fcluster(z, distance, criterion='distance')

    names = list([str(n) for n in names]) if names is not None else [f'cell_{ii}' for ii in range(len(cluster_id))]

    for merge in z:
        # print(merge)
        if merge[2] > distance:
            break
        ii, jj = int(merge[0]), int(merge[1])
        
        if ii < len(names):
            # first leaf is a single data set
            name0 = names[int(ii)]
            cid = cluster_id[ii]
        else:
            # first leaf is a merged data set
            name0 = merge_list[int(ii)-len(names)]
            cid = cluster_list[int(ii)-len(names)]
            
        if jj < len(names):
            name1 = names[int(jj)]
            assert cluster_id[jj] == cid    # make sure that both leaves are from the same cluster
        else:
            name1 = merge_list[int(jj)-len(names)]
            assert cluster_list[int(jj)-len(names)] == cid
            
        merge_list.append((name0, name1))
        cluster_list.append(cid)
        
    return merge_list, cluster_list


def flatten_to_str(in_names: Union[Tuple[Union[Tuple, str], Union[Tuple, str]], str], sep: str = ':') -> str:
    """Generates unique string identifiers from a nested tuple of strings as returned by `build_merge_tree`"""
    return in_names if isinstance(in_names, str) else sep.join([flatten_to_str(fn, sep) for fn in in_names])