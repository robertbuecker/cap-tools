from utils import get_clusters, parse_cap_csv, put_in_order, to_radian, to_sin, unit_cell_lcv_distance, volume, volume_difference, write_cap_csv
import numpy as np
import yaml
from scipy.cluster.hierarchy import linkage
import csv
import io
from collections import namedtuple
from typing import Dict, Optional


class CellList:

    def __init__(self, cells: np.ndarray, ds: Optional[dict] = None, weights: Optional[np.ndarray] = None):
        self._cells = put_in_order(cells)
        self._weights = np.array([1]*cells.shape[0]) if weights is None else weights
        if ds is None:
            self.ds = []
            for c in cells:
                self.ds.append({'unit cell': ' '.join(list(c))})
        else:
            self.ds = ds

    def __len__(self):
        return self._cells.shape[0]

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

    def to_csv(self, fn: str):
        write_cap_csv(fn, self.ds)

    def cluster(self,
                 distance: float=None,
                 method: str="average",
                 metric: str="euclidean",
                 use_radian: bool=False,
                 use_sine: bool=False) -> Dict[int,'CellList']:
                """Perform hierarchical cluster analysis on a list of cells. 

                method: lcv, volume, euclidean
                distance: cutoff distance, if it is not given, pop up a dendrogram to
                    interactively choose a cutoff distance
                use_radian: Use radian instead of degrees to downweight difference
                use_sine: Use sine for unit cell clustering (to disambiguousize the difference in angles)
                """

                from scipy.spatial.distance import pdist

                if use_sine:
                    _cells = to_sin(self.cells)
                elif use_radian:
                    _cells = to_radian(self.cells)
                else:
                    _cells = self.cells

                if metric.lower() == "lcv":
                    dist = pdist(_cells, metric=unit_cell_lcv_distance)
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
                print("")

                clusters_idx = get_clusters(z, self.cells, distance=distance)

                clusters = {}
                for k, cluster_idx in clusters_idx.items():
                    clusters[k] = CellList(cells = self.cells[cluster_idx],
                                           ds=[d for ii, d in enumerate(self.ds) if ii in cluster_idx],
                                           weights=self.weights[cluster_idx])

                return clusters, z