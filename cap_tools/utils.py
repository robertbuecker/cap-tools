from collections import defaultdict, namedtuple
import csv
from math import radians, cos, floor, log10
from pathlib import Path
from typing import *
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
import sys
import subprocess
import os
import warnings


class TextRedirector(object):
    # from:
    # https://stackoverflow.com/questions/12351786/how-to-redirect-print-statements-to-tkinter-text-widget
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, string):
        self.widget.configure(state="normal")
        self.widget.insert("end", string, (self.tag,))
        self.widget.configure(state="disabled")

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

def get_version():
                
        try:
            result = subprocess.run(['git', 'describe'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                # we are inside a working git repository
                return result.stdout.strip()                            
        except Exception as e:
            pass
        
        try:       
            # we are inside a PyInstaller executable
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
            
        try:
            with open(os.path.join(base_path, 'version.txt'), 'r') as fh:
                return fh.read().strip()
            
        except Exception as e:
            return('Could not determine version')


def niggli_to_lattice(a, b, c, alpha, beta, gamma):
    """
    Convert Niggli cell parameters (a, b, c, alpha, beta, gamma) to a 3x3 lattice matrix.
    """
    # Convert angles from degrees to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    # Calculate the lattice vectors
    A = np.zeros((3, 3))

    A[0, 0] = a
    A[1, 0] = b * np.cos(gamma)
    A[1, 1] = b * np.sin(gamma)
    A[2, 0] = c * np.cos(beta)
    A[2, 1] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    A[2, 2] = c * np.sqrt(1 - np.cos(beta)**2 - (np.cos(alpha) - np.cos(beta) * np.cos(gamma))**2 / np.sin(gamma)**2)

    return A


def spacegroup_to_bravais(number):
    """
    Map spacegroup number (1–230) to extended Bravais lattice type.
    """
    if number in range(1, 3):
        return 'aP'  # Simple cubic (P)
    elif number in range(3, 16):
        return 'mP'  # Monoclinic (P)
    elif number in range(16, 75):
        return 'oP'  # Orthorhombic (P)
    elif number in range(75, 89):
        return 'tP'  # Tetragonal (P)
    elif number in range(89, 143):
        return 'tI'  # Tetragonal (I)
    elif number in range(143, 149):
        return 'hR'  # Hexagonal (R)
    elif number in range(149, 168):
        return 'hP'  # Hexagonal (P)
    elif number in range(168, 195):
        return 'oF/oI/oC'  # Some ambiguity (for cubic/face-centered/centered)
    elif number in range(195, 207):
        return 'cP'  # Cubic primitive
    elif number in range(207, 225):
        return 'cF'  # Cubic face-centered
    elif number in range(225, 231):
        return 'cI'  # Cubic body-centered
    else:
        return f'Unknown ({number})'  # Not a valid space group number

def identify_bravais_lattice_from_niggli(a, b, c, alpha, beta, gamma, symprec=1e-4):
    
    # Convert Niggli parameters to lattice matrix
    lattice = niggli_to_lattice(a, b, c, alpha, beta, gamma)
    print(lattice)

    positions = np.array([[0.0, 0.0, 0.0]])
    numbers = [1]
    cell = (lattice, positions, numbers)

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        return {"error": "Symmetry could not be determined."}

    # Map spacegroup number to Bravais lattice
    bravais = spacegroup_to_bravais(dataset.number)

    return bravais, \
        {
        "transformation_matrix": dataset.transformation_matrix,
        "origin_shift": dataset.origin_shift,
        "bravais_lattice": bravais,
        "hall_symbol": dataset.hall,
        "international_symbol": dataset.international,
    }

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
        
    key = 'Current unit cell' if use_raw_cell else 'Final SG unit cell'
    ds = [d for d in ds if d[key]] if filter_missing else ds # filter experiments with empty cell parameters (not indexed)
    cells = np.array([d[key].split() for d in ds]).astype(float)
    weights = np.array([1 for d in ds])
    
    def get_centring(d) -> str:
        if use_raw_cell:
            ctr = d['Current lattice'].strip()[-1]
        else:
            ctr = d['Space group RED'].strip()[0]
        if ctr not in ['P', 'I', 'F', 'A', 'B', 'C', 'R', 'H']:
            warnings.warn(RuntimeWarning(f'Unknown centring {ctr}. Assuming P'))
            ctr = 'P'
        return ctr
    
    centrings = np.array([get_centring(d) for d in ds])

    return ds, cells, weights, centrings


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

ClusterOptions = namedtuple('ClusterOptions', ['preproc', 'metric', 'method', 'centring'])

def node_id_from_link(Z):
    # get list of node IDs (i.e., positions of entries in the linkage matrix) ordered like the
    # link items returned by the dendrogram function. Adapted from:
    # https://stackoverflow.com/questions/73103010/matching-up-the-output-of-scipy-linkage-and-dendrogram
        
    def append_index(n, i, node_id_list):
        
        # i is the ID of the node (counting in all 2 * n - 1 nodes)
        # so i-n is the idx in the "Z"
        if i < n:
            return
        aa = int(Z[i - n, 0])
        ab = int(Z[i - n, 1])

        append_index(n, aa, node_id_list)
        append_index(n, ab, node_id_list)

        node_id_list.append(i-n)
        # Imitate the progress in hierarchy.dendrogram
        # so how `i-n` is appended , is the same as how the element in 'icoord'&'dcoord' be.
        return    
        
    n = Z.shape[0] + 1
    i = 2 * n - 2
    node_id_list = []
    append_index(n, i, node_id_list)

    return node_id_list