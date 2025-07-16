from collections import defaultdict, namedtuple
import configparser
import csv
from glob import glob
import hashlib
from math import radians, cos, floor, log10
from pathlib import Path
from time import sleep
from typing import *
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
import sys
import subprocess
import os
import warnings
import xml.etree.ElementTree as ET

import pandas as pd


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

def parse_cap_meta(experiments: Union[str, List[str]], 
                   log_fun: Optional[Callable[[str], None]] = None,
                   include_merged: bool = False,
                   exclude: List[str] = ('tomo', 'DD_Calib', 'Preset', 'Cluster'),
                   include: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    
    if isinstance(experiments, str):
        experiments = [experiments]
    
    log = print if log_fun is None else log_fun
    
    # Step 1: get full path names of all experiments in a robust and consistent way
    exp_list = []
    for exp_entry in experiments:

        if not os.path.exists(exp_entry):
            raise FileNotFoundError(f'Input folder or CSV file {exp_entry} does not exist')

        log(f'Scanning {exp_entry}...')
        if exp_entry.endswith('.csv'):
            with open(exp_entry) as fh:
                for _ in range(7):
                    _ = fh.readline()
                ds = list(csv.DictReader(fh))
                new_exp = [os.path.join(d['Dataset path'], d['Experiment name']) for d in ds]
                log(f'Found {len(new_exp)} experiments in {exp_entry}')

        else:            
            new_exp = [os.path.splitext(fn)[0] for fn in glob(os.path.join(exp_entry,'**\\*.par'), recursive=True) if ('_cracker' not in fn)]
            log(f'Found {len(new_exp)} .par files in {exp_entry}')
            
        exp_list.extend(new_exp)
        
    N = len(exp_list)
    exp_list = [fn for fn in exp_list if all([(expr not in fn) for expr in exclude])]
    log(f'{N-len(exp_list)} experiments were excluded by the exclude list {exclude}')
    
    N = len(exp_list)
    if include is not None:
        exp_list = [fn for fn in exp_list if any([(expr in fn) for expr in include])]
        log(f'{N-len(exp_list)} experiments were excluded by the include list {include}')

    exp_list = sorted(list(set(exp_list)))
    log(f'Found {len(exp_list)} experiments of correct type')
    
    # Step 2: iterate through all experiments and collate metadata from various files
    info = []
    
    for ii, exp in enumerate(exp_list):
        
        # STEP 2.1: XML info file
        info_fn = os.path.join(os.path.dirname(exp), 'experiment_results.xmlinfo')
        if not os.path.exists(info_fn):
            log(f'WARNING: {info_fn} is missing. Skipping this experiment.')
            continue
        
        info_xml_str = open(info_fn).read()
        tree = ET.fromstring('<root>\n' + info_xml_str + '\n</root>')
        exp_info = {'path': exp}
        
        if (exp_type := tree.find('__EXPERIMENT_INFO__/__EXPERIMENT_TYPE__')) is not None:
            if (float(exp_type.text) == 6.) and not include_merged:
                log(f'Skipping merged experiment {exp} (type 6)')
                continue
        
        # generate hash digest stable information (not changing with reprocessing or moving)
        if (user := tree.find('__EXPERIMENT_INFO__/__USER__')) is not None:
            user = user.text
        else:
            user = 'anonymous'
            
        if (exp_time := tree.find('__EXPERIMENT_INFO__/__START_TIME__')) is not None:
            exp_time = exp_time.text
        else:
            exp_time = 'unknown time'
            
        if (exp_name := tree.find('__EXPERIMENT_INFO__/__EXPERIMENT_PAR_NAME_WOEXT__')) is not None:
            exp_name = exp_name.text
        else:
            exp_name = os.path.basename(exp)
        
        exp_info['name'] = exp_name
        
        xml_entries = {
            'scan_range': tree.find('__EXPERIMENT_INFO__/__SCAN_RANGE__'),
            'detector_distance': tree.find('__EXPERIMENT_INFO__/__DETECTOR_DISTANCE__'),
            'indexation': tree.find('__EXPERIMENT_RESULTS__/__INDEXATION__'),
            'e1': tree.find('__EXPERIMENT_RESULTS__/__MOSAICITY__/__MOSAICITY_E1__'),
            'e2': tree.find('__EXPERIMENT_RESULTS__/__MOSAICITY__/__MOSAICITY_E2__'),
            'e3': tree.find('__EXPERIMENT_RESULTS__/__MOSAICITY__/__MOSAICITY_E3__'),
            'diff_limit': tree.find('__EXPERIMENT_RESULTS__/__DIFFLIMIT__'),
            'r_int': tree.find('__EXPERIMENT_RESULTS__/__RINT__'),
            'stage_x': tree.find('__EXPERIMENT_INFO__/__STAGE_POSITION__/__STAGE_POSITION_X__'),
            'stage_y': tree.find('__EXPERIMENT_INFO__/__STAGE_POSITION__/__STAGE_POSITION_Y__'),
            'stage_z': tree.find('__EXPERIMENT_INFO__/__STAGE_POSITION__/__STAGE_POSITION_Z__')
        }
        
        for k, v in xml_entries.items():
            if v is not None:
                exp_info[k] = float(v.text)
        
        m = hashlib.md5()
        hash_text = ';'.join([user, exp_time, exp_name])
        m.update(hash_text.encode()) # this line defines what gets hashed
        exp_info['digest'] = m.hexdigest()
                
        # STEP 2.2: queue info file
        xpos, ypos = 387.5, 192.5
        if os.path.exists(qedfile := os.path.join(os.path.dirname(exp), 'metadataexpsettings.qed')):
            with open(qedfile, 'r') as fh:
                for ln in fh:
                    if ln.startswith('s_experimentsettings_metadata.s_queue_singletask.ssampleinfo_template.srequestedposition.la_abspixelrequestedposition_xy2[0]'):
                        xpos = int(ln.split('=')[1].strip())
                    if ln.startswith('s_experimentsettings_metadata.s_queue_singletask.ssampleinfo_template.srequestedposition.la_abspixelrequestedposition_xy2[1]'):
                        ypos = int(ln.split('=')[1].strip())

        exp_info['grain_x_px'] = xpos
        exp_info['grain_y_px'] = ypos
        
        # STEP 2.3: data collection ini file
        try:
            config = configparser.ConfigParser()
            config.read(os.path.join(os.path.dirname(exp), 'expinfo', exp_name + '_datacoll.ini'))
            OL_demag, visual_pxs = 47, 0.036
            exp_info['aperture_px'] = float(config['MicroED'].get('Aperture SA info', None)) / OL_demag / visual_pxs
        except Exception as err:
            log('Could not decode aperture size for', exp)
            
        # STEP 2.4: check grain images and diffraction images
        extensions = ['rodhypix', 'jpg', 'tiff']
        kinds = {'diff': 'middle_microed_diff_snapshot',
                 'grain': 'microed_grain_snapshot',
                 'minimap': 'microed_minimap_snapshot',
                 'post': 'microed_post_snapshot'}
        
        #TODO minimap JPG needs special treatment, as it has coordinates in filename
        for kind, fn_label in kinds.items():
            for ext in extensions:
                fn = os.path.join(os.path.dirname(exp), exp_name + '_' + fn_label + '.' + ext)
                if os.path.exists(fn):
                    exp_info[kind + '-' + ext] = fn
                    
        info.append(exp_info)
        
    return info

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