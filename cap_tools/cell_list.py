from .utils import get_clusters, parse_cap_csv, put_in_order, \
    to_radian, to_sin, unit_cell_lcv_distance, volume, volume_difference, write_cap_csv, build_merge_tree, flatten_to_str
import numpy as np
import yaml
from scipy.cluster.hierarchy import linkage
import csv
import io, os
from collections import namedtuple
from typing import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class CellList:

    def __init__(self, cells: np.ndarray, ds: Optional[dict] = None, weights: Optional[np.ndarray] = None, merge_tree: Optional[list] = None):
        self._cells = put_in_order(cells)
        self._weights = np.array([1]*cells.shape[0]) if weights is None else weights
        self._merge_tree = [] if merge_tree is None else merge_tree
        self.pca_whiten = False # somewhat redundant if SEuclidean is available as clustering metric

        if ds is None:
            self.ds = []
            for c in cells:
                self.ds.append({'unit cell': ' '.join(list(c))})
        else:
            self.ds = ds

    def __len__(self):
        return self._cells.shape[0]
    
    @property
    def merge_tree(self):
        return self._merge_tree

    @property
    def cells(self):
        return self._cells

    @property
    def weights(self):
        return self._weights

    @property
    def volumes(self):
        return np.array([volume(cell) for cell in self.cells])

    @property
    def stats(self):
        cdat = np.concatenate([self.cells, self.volumes.reshape(-1,1)], axis=1)
        CellStats = namedtuple('CellStats', ['mean', 'std', 'min', 'max'])
        return CellStats(np.mean(cdat, axis=0),
                np.std(cdat, axis=0),
                np.min(cdat, axis=0),
                np.max(cdat, axis=0))
        
    @property
    def cells_standardized(self):
        return StandardScaler().fit_transform(self.cells)
    
    @property
    def cells_pca(self):
        """PCA-filtered normalized cell parameters"""
        return PCA(whiten=self.pca_whiten).fit_transform(self.cells)
    
    @property
    def G6(self):
        """G6 metric (Gruber parametrization)"""
        cos = lambda al: np.cos(al*np.pi/180)
        return np.stack([(c[0]**2, c[1]**2, c[2]**2, 
                          2*cos(c[3])*c[1]*c[2], 2*cos(c[4])*c[2]*c[0], 2*cos(c[5])*c[0]*c[1]) 
                         for c in self.cells], axis=0)
        
    @property
    def diagonals(self):
        """Facet diagonals (as used for LCV, see Foadi et al. 2013)"""
        return np.stack([(
            np.sqrt(c[0]**2 + c[1]**2 - 2*c[0]*c[1]*np.cos(np.radians(180 - c[5]))),
            np.sqrt(c[2]**2 + c[0]**2 - 2*c[2]*c[0]*np.cos(np.radians(180 - c[4]))),
            np.sqrt(c[1]**2 + c[2]**2 - 2*c[1]*c[2]*np.cos(np.radians(180 - c[3])))
        ) for c in self.cells], axis=0)
        
    @property
    def diagonals_PCA(self):
        """PCA-filtered normalized facet diagonals"""
        return PCA(whiten=self.pca_whiten).fit_transform(self.diagonals)

    @property
    def table(self):
        sh = io.StringIO()
        writer = csv.DictWriter(sh, fieldnames=self.ds[0].keys())
        writer.writeheader()
        writer.writerows(self.ds)
        return sh.getvalue()

    @classmethod
    def from_yaml(cls, fn, use_raw_cell=True):
        ds = yaml.load(open(fn, "r"), Loader=yaml.Loader)
        key = "raw_unit_cell" if use_raw_cell else "unit_cell"
        # prune based on NaNs (missing cells)
        ds = [d for d in ds if not any(np.isnan(d[key]))]
        cells = np.array([d[key] for d in ds])
        weights = np.array([d["weight"] for d in ds])

    @classmethod
    def from_csv(cls, fn, use_raw_cell=True):
        ds, cells, weights = parse_cap_csv(fn, use_raw_cell, filter_missing=True)
        return cls(cells=cells, ds=ds)
    
    def get_merging_paths(self, prefix: str = '', common_path: Optional[str] = None, short_form: bool = False,
                          appendices: Union[list, tuple] = ('', '_autored', '_auto')):
        """
        define merging paths: by default, place merged file into folder of involved experiment with lowest number        
        """
        
        exps = {} # locate non-merged proffit files of single experiments
        for name, path in ((d['Experiment name'], d['Dataset path']) for d in self.ds):
            for appendix in appendices:
                if os.path.exists(os.path.join(path, f'{name}{appendix}.rrpprof')):
                    exps[name] = (path, f'{name}{appendix}')
                    break
            else:
                print(f'No rrpprof file found for {name} in {path} - Skipping.')
                
        # create mangle-able ID strings from merging tree
        out_codes = [flatten_to_str(mt) for mt in self.merge_tree]
        in_codes = [(flatten_to_str(mt[0]), flatten_to_str(mt[1])) for mt in self.merge_tree]
        sep = '-'
        
        if short_form:
            out_paths = [os.path.join(exps[sorted(oc.split(':'))[0]][0] if common_path is None else common_path, 
                                  '-'.join([prefix, f'ID{ii+1:03d}', f'{len(oc.split(":"))}exp'])) 
                    for ii, oc in enumerate(out_codes)]
        else:        
            out_paths = [os.path.join(exps[sorted(fn.split(':'))[0]][0] if common_path is None else common_path, 
                                  '-'.join([prefix, fn]).replace(':', sep)) 
                    for fn in out_codes]
        
        # get info about which initial proffit files are in each output file
        # this corresponds to the out_code string ID, but with naked exp filenames replaced by the 
        # finalization name (to be fully unique) and sorted identifiers
        out_info = [
            ':'.join([exps[in_code][1] for in_code in sorted(out_code.split(':'))])
                    for out_code in out_codes]

        # get input paths
        in_paths = []
        for in_code in in_codes:
            paths = []
            for cd in in_code:
                if not ':' in cd:
                    # non-merged file
                    paths.append(os.path.join(*exps[cd]))
                else:
                    # merged file
                    paths.append(out_paths[out_codes.index(cd)])            
            in_paths.append(tuple(paths))
            
        return out_paths, in_paths, out_codes, out_info

    def to_csv(self, fn: str):
        write_cap_csv(fn, self.ds)

    def cluster(self,
                 distance: Optional[float]=None,
                 method: str="average",
                 metric: str="euclidean",
                 preproc: str="none") -> Dict[int,'CellList']:
                """Perform hierarchical cluster analysis on a list of cells. 

                method: lcv, volume, euclidean
                distance: cutoff distance, if it is not given, pop up a dendrogram to
                    interactively choose a cutoff distance
                use_radian: Use radian instead of degrees to downweight difference
                use_sine: Use sine for unit cell clustering (to disambiguousize the difference in angles)
                """

                from scipy.spatial.distance import pdist
                
                pp = preproc.lower()

                # cell data conditioning
                if pp == 'none':
                    _cells = self.cells
                elif pp == 'standardized':
                    _cells = self.cells_standardized
                elif pp == 'pca':
                    _cells = self.cells_pca
                elif pp == 'sine':
                    _cells = to_sin(self.cells)
                elif pp == 'radians':
                    _cells = to_radian(self.cells)
                elif pp == 'diagonals':
                    _cells = self.diagonals
                elif pp == 'diagonalspca':
                    _cells = self.diagonals_PCA
                elif pp == 'g6':
                    _cells = self.G6
                else:
                    raise ValueError(f'Unknown preprocessing method {preproc}')

                # cell distance metric
                if metric.lower() == "lcv":
                    dist = pdist(_cells, metric=unit_cell_lcv_distance)
                    z = linkage(dist,  method=method)
                    distance = round(0.5*max(z[:,2]), 4) if distance is None else distance
                elif metric.lower() == "alcv":
                    dist = pdist(_cells, metric=lambda cell1, cell2: unit_cell_lcv_distance(cell1, cell2, True))
                    z = linkage(dist,  method=method)
                    distance = round(0.5*max(z[:,2]), 4) if distance is None else distance                    
                elif metric.lower() == "volume":
                    dist = pdist(_cells, metric=volume_difference)
                    z = linkage(dist,  method=method)
                    distance = 250.0 if distance is None else distance
                else:
                    z = linkage(_cells,  metric=metric, method=method.lower())
                    distance = 2.0 if distance is None else distance

                # if not distance:
                #     distance = distance_from_dendrogram(z, ylabel=metric, initial_distance=initial_distance, labels=labels)

                print(f"Linkage method = {method}")
                print(f"Cutoff distance = {distance}")
                print(f"Distance metric = {metric}")
                print(f"Preprocessing = {preproc}")
                print("")

                clusters_idx = get_clusters(z, self.cells, distance=distance)
                try:
                    names = [d['Experiment name'] for d in self.ds]
                except KeyError:
                    names = [str[ii] for ii in range(len(self))]
                    
                self._merge_tree, cid = build_merge_tree(z, distance=distance, names=names)

                clusters = {}
                for k, cluster_idx in clusters_idx.items():
                    clusters[k] = CellList(cells = self.cells[cluster_idx],
                                           ds=[d for ii, d in enumerate(self.ds) if ii in cluster_idx],
                                           weights=self.weights[cluster_idx],
                                           merge_tree=[mt for ii, mt in zip(cid, self.merge_tree) if ii==k])

                return clusters, z