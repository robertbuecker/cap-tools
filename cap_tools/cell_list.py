import warnings
from .utils import parse_cap_csv, order_uc_pars, \
    unit_cell_lcv_distance, volume, volume_difference, write_cap_csv, ClusterOptions
import numpy as np
from scipy.cluster.hierarchy import linkage
import csv
import io, os
from collections import namedtuple
from typing import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict


class CellList:

    def __init__(self, cells: np.ndarray, ds: Optional[dict] = None, weights: Optional[np.ndarray] = None,
                 centrings: Optional[np.ndarray] = None, reduce: bool = False,
                 merge_tree: Optional[Tuple] = None, linkage_z: Optional[np.ndarray] = None, 
                 cluster_pars: Optional[ClusterOptions] = None, cluster_distance: Optional[float] = None):
        
        if reduce:
            # reduce cells to Niggli form
            import gemmi 
            cells_reduced = []

            for cell, centring in zip(cells, centrings):
                uc = gemmi.UnitCell(*cell)
                gv = gemmi.GruberVector(uc, centring=centring)
                gv.niggli_reduce()
                cells_reduced.append(gv.get_cell().parameters)
                
            self._cells = np.array(cells_reduced)
        else:
            # no reduction: order the cell parameters according to axis lengths
            # TODO this leads to wrong centrings for A, B, C centered cells!
            self._cells = order_uc_pars(cells)
                
        self._weights = np.array([1]*cells.shape[0]) if weights is None else weights
        self._centrings = np.array(['P']*cells.shape[0]) if ((centrings is None) or reduce) else centrings
        
        # clustering parameters
        self._cluster_pars = cluster_pars
        self._distance: Optional[float] = cluster_distance
        self._merge_tree: Tuple[Union[Tuple, str], Union[Tuple, str]] = merge_tree
        self._z: Optional[np.ndarray] = linkage_z
        self._clusters: Dict[int, CellList] = {}

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
    def centrings(self):
        return self._centrings
    
    @property
    def centring(self):
        if len(ctrs := np.unique(self.centrings)) > 1:
            return '?'
        else:
            return ctrs[0]

    @property
    def weights(self):
        return self._weights

    @property
    def volumes(self):
        return np.array([volume(cell) for cell in self.cells])

    @property
    def stats(self):
        cdat = np.concatenate([self.cells, self.volumes.reshape(-1,1)], axis=1)
        CellStats = namedtuple('CellStats', ['mean', 'std', 'min', 'max', 'centring'])
        if len(ctrs := np.unique(self.centrings)) > 1:
            warnings.warn(f'Warning: {len(ctrs)} different centrings found in the dataset ({", ".join(ctrs)}). Not computing statistics.')
            dummy_value = np.nan*np.ones(cdat.shape[1])            
            return CellStats(dummy_value,
                             dummy_value,
                             dummy_value,
                             dummy_value,
                             np.array('?'))
        else:
            return CellStats(np.mean(cdat, axis=0),
                    np.std(cdat, axis=0),
                    np.min(cdat, axis=0),
                    np.max(cdat, axis=0),
                    ctrs[0])
        
    @property
    def cells_standardized(self):
        return StandardScaler().fit_transform(self.cells)
    
    @property
    def cells_pca(self):
        """PCA-filtered normalized cell parameters"""
        return PCA(whiten=False).fit_transform(self.cells)
    
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
        return PCA(whiten=False).fit_transform(self.diagonals)

    @property
    def table(self):
        sh = io.StringIO()
        writer = csv.DictWriter(sh, fieldnames=self.ds[0].keys())
        writer.writeheader()
        writer.writerows(self.ds)
        return sh.getvalue()

    @classmethod
    def from_csv(cls, fn, use_raw_cell=True, reduce=False) -> 'CellList':
        ds, cells, weights, centrings = parse_cap_csv(fn, use_raw_cell, filter_missing=True)
        return cls(cells=cells, ds=ds, centrings=centrings, weights=None, reduce=reduce)
    
    #TODO add a from_cluster_result option

    def to_csv(self, fn: str):
        write_cap_csv(fn, self.ds)                   

    def cluster(self,
                 distance: Optional[float]=None,
                 cluster_pars: Optional[ClusterOptions] = None) -> Dict[int,'CellList']:
                """Perform hierarchical cluster analysis on a list of cells. 

                method: lcv, volume, euclidean
                distance: cutoff distance, if it is not given, pop up a dendrogram to
                    interactively choose a cutoff distance
                centring: 'ignore' (default), 'split'
                preproc: data preprocessing: none, standardized, pca, diagonals, diagonalspca, g6
                z: linkage matrix. If provided, no recomputation is performed and method/metric/preproc are ignored
                """

                # if (centring != 'ignore') and cluster_pars.preproc.lower() != 'none':
                #     raise NotImplementedError('Centring is not yet compatible with preprocessing.')                    
                
                from scipy.spatial.distance import pdist, squareform
                
                if (cluster_pars is None) and self._cluster_pars is None:
                    return ValueError('First clustering run; you need to supply parameters.')
                elif (cluster_pars is None) or (cluster_pars == self._cluster_pars):
                    z = self._z
                    if distance is None: distance = self._distance
                else:
                    # pairwise distances and linkage matrix need to be recomputed
                    
                    # TODO: prevent clustering of cells with different centring
                                    
                    # cell data conditioning
                    pp_methods = {'none': self.cells, 
                                'standardized': self.cells_standardized, 
                                'pca': self.cells_pca,
                                'diagonals': self.diagonals,
                                'diagonalspca': self.diagonals_PCA,
                                'g6': self.G6}
                    try: 
                        _cells = pp_methods[cluster_pars.preproc.lower()]
                    except IndexError:
                        raise ValueError(f'Unknown preprocessing method {cluster_pars.preproc}')
                    
                    meth = cluster_pars.method.lower()
                    centring = cluster_pars.centring.lower()

                    def pdist_centring(cells, *args, **kwargs):
                        """Compute pairwise distances between cells, taking into account the centring"""
                        
                        if (meth != 'average') and (cluster_pars.metric.lower() not in ['euclidean', 'volume', 'lcv', 'alcv']):
                            raise ValueError('For the selected metric, only `average` is supported as method.')                        
                        
                        dist = pdist(cells, *args, **kwargs)                        
                        if centring == 'ignore':
                            return dist
                        
                        centrings = self.centrings
                        pd_square = squareform(dist)
                        maxval = np.max(pd_square.ravel())*1.05
                        for i in range(len(centrings)):
                            for j in range(i+1, len(centrings)):
                                if centrings[i] != centrings[j]:
                                    pd_square[i,j] = pd_square[j,i] = maxval
                        return squareform(pd_square)
                    
                    try:
                        # compute distances and linkage
                        if cluster_pars.metric.lower() == "lcv":
                            dist = pdist_centring(_cells, metric=unit_cell_lcv_distance)
                            z = linkage(dist,  method=meth, optimal_ordering=True)
                            distance = round(0.5*max(z[:,2]), 4) if distance is None else distance
                        elif cluster_pars.metric.lower() == "alcv":
                            dist = pdist_centring(_cells, metric=lambda cell1, cell2: unit_cell_lcv_distance(cell1, cell2, True))
                            z = linkage(dist,  method=meth, optimal_ordering=True)
                            distance = round(0.5*max(z[:,2]), 4) if distance is None else distance                    
                        elif cluster_pars.metric.lower() == "volume":
                            dist = pdist_centring(_cells, metric=volume_difference)
                            z = linkage(dist, method=meth, optimal_ordering=True)
                            distance = 250.0 if distance is None else distance
                        else:
                            dist = pdist_centring(_cells, metric=cluster_pars.metric.lower())
                            z = linkage(dist,  method=meth, optimal_ordering=True)
                            distance = 2.0 if distance is None else distance
                            
                        self._cluster_pars = cluster_pars
                         
                    except Exception as err:
                        print(f'Could not recompute dendrogram with parameters:')                        
                        print(f"-> Preprocessing = {cluster_pars.preproc}")
                        print(f"-> Distance metric = {cluster_pars.metric}")
                        print(f"-> Linkage method = {cluster_pars.method}")
                        print(f"-> Centring handling = {cluster_pars.centring}")
                        print(f'The error was:, {str(err)}')
                        raise err

                    # if not distance:
                    #     distance = distance_from_dendrogram(z, ylabel=metric, initial_distance=initial_distance, labels=labels)
                    print(f"Recomputing dendrogram")
                    print(f"-> Preprocessing = {cluster_pars.preproc}")
                    print(f"-> Distance metric = {cluster_pars.metric}")
                    print(f"-> Linkage method = {cluster_pars.method}")
                    print(f"-> Centring handling = {cluster_pars.centring}")
                    

                    self._z = z
                    
                self._distance = distance
                    
                print(f"Reclustering with distance = {distance}")
                cluster = fcluster(z, distance, criterion='distance')
                
                # reformat linkage matrix into a recursive-tuple "merge tree"
                try:
                    names = [d['Experiment name'] for d in self.ds]
                except KeyError:
                    names = [str[ii] for ii in range(len(self))]
                    
                self._merge_tree = []
                merge_tree_cids = []
                node_cids = []

                for node in z:

                    if node[2] > distance:
                        # node is not part of a cluster
                        node_cids.append(-1)
                        continue
                    
                    ii, jj = int(node[0]), int(node[1])
                    
                    if ii < len(names):
                        # first leaf is a single data set
                        name0 = names[int(ii)]
                        cid = cluster[ii]
                    else:
                        # first leaf is a merged data set
                        name0 = self._merge_tree[int(ii)-len(names)]
                        cid = merge_tree_cids[int(ii)-len(names)]
                        
                    if jj < len(names):
                        name1 = names[int(jj)]
                        assert cluster[jj] == cid    # make sure that both leaves are from the same cluster
                    else:
                        name1 = self._merge_tree[int(jj)-len(names)]
                        assert merge_tree_cids[int(jj)-len(names)] == cid
                        
                    self._merge_tree.append((name0, name1))
                    merge_tree_cids.append(cid)
                    node_cids.append(cid)
                
                node_cids = np.array(node_cids)
                
                # make cell list for each cluster, without having to recompute linkage parameters for each
                cluster_lists = {}
                for cid in np.unique(cluster):          
                              
                    idcs = np.flatnonzero(cluster == cid)                   
                    if len(idcs) == 1:
                        # sort out singletons
                        continue                   
                    
                    cluster_lists[cid] = CellList(cells = self.cells[idcs],
                                           ds=[d for ii, d in enumerate(self.ds) if ii in idcs],
                                           weights=self.weights[idcs],
                                           centrings=self.centrings[idcs],
                                           merge_tree=[mt for ii, mt in zip(merge_tree_cids, self.merge_tree) if ii==cid],
                                           linkage_z=z[node_cids == cid, :],
                                           cluster_pars=self._cluster_pars,
                                           cluster_distance=distance)

                self._clusters = cluster_lists
                
                return node_cids
            
    def save_clusters(self, fn_template: str, list_fn: Optional[str] = None, out_dir: str = None,
                      selection: List[int] = (), top_only: bool = False):
        
        ver = 1
        info_fn = os.path.splitext(fn_template)[0] + '_cluster_info.csv'
        cluster_data_folder = os.path.dirname(info_fn)
        
        os.makedirs(cluster_data_folder, exist_ok=True)
        
        with open(info_fn, 'w') as ifh:
            ifh.write(
                f'VERSION {ver}\n'
                f'HEADER INFO:\n'
                f'Experiment list: {list_fn if list_fn is not None else "(unknown)"}\n'
                f'Preprocessing: {self._cluster_pars.preproc}\n'
                f'Metric: {self._cluster_pars.metric}\n'
                f'Method: {self._cluster_pars.method}\n'
                f'Distance: {self._distance}\n'
            )
            ifh.write('Name,File path,Cluster,Data sets,Merge code\n')
            
            for ii, (c_id, cluster) in enumerate(self.clusters.items()):
                if c_id not in selection:
                    print(f'Skipping Cluster {c_id} (not in selection)')
                    continue
                try:
                    out_paths, in_paths, out_codes, out_info = cluster.get_merging_paths(prefix=f'C{c_id}', 
                                                                                     short_form=True, 
                                                                                     legacy=False, 
                                                                                     common_path=os.path.join(cluster_data_folder, f'Cluster-{c_id}'))  
                except FileNotFoundError as err:
                    print('WARNING: ', str(err) + f' - SKIPPING CLUSTER {c_id}')
                    continue
      
                N_exp = [len(inp) for inp in in_paths]
                full_merge_pos = N_exp.index(max(N_exp))
      
                for ii, (out, inp, code, info) in enumerate(zip(out_paths, in_paths, out_codes, out_info)):
                    if (not top_only) or (ii == full_merge_pos):
                        out_path = os.path.join(out, os.path.basename(out)) if out_dir else ''                 
                        ifh.write(f'{os.path.basename(out)},{out_path},{c_id},{' | '.join(inp)},{code}\n')
                    
                cluster_fn = os.path.splitext(fn_template)[0] + f'-Cluster-{c_id}.csv'
                cluster.to_csv(cluster_fn)
                print(f'Wrote cluster {c_id} with {len(cluster)} crystals to file {cluster_fn}')
                
        return info_fn
        
    @property
    def clusters(self) -> Dict[int, 'CellList']:
        return self._clusters
            
    def get_merge_codes(self, sep: str = ':') -> Tuple[List[str], List[Tuple[str, str]]]:
        """Generates unique string ID codes for each merge node containing all dataset names in that merge node
        separated by a definable separator. Also, for each merge node, generates a pair of string IDs corresponding
        to the two child nodes directly merged into that node.

        Args:
            sep (str, optional): Separator. Defaults to ':'.

        Returns:
            out_codes, in_codes: Lists of string ID codes for each merge node, and pairs of ID codes of the child nodes
        """
        
        def flatten_to_str(in_names: Union[Tuple[Union[Tuple, str], Union[Tuple, str]], str], sep=sep) -> str:    
            # recursive function to flatten merging Tuples
            return in_names if isinstance(in_names, str) else sep.join([flatten_to_str(fn, sep) for fn in in_names])
        
        out_codes = [flatten_to_str(mt) for mt in self.merge_tree]
        in_codes = [((flatten_to_str(mt[0]), flatten_to_str(mt[1])) if isinstance(mt, tuple) else (None, None)) 
                    for mt in self.merge_tree] # needs to account for singletons
        
        return out_codes, in_codes
            
    def get_merging_paths(self, prefix: str = '', common_path: Optional[str] = None, short_form: bool = False,
                          appendices: Union[list, tuple] = ('', '_autored', '_auto'), legacy: bool = True):
        """
        define merging paths: by default, place merged file into folder of involved experiment with lowest number        
        """
        
        exps = {} # dict of Tuple[path, finalization_name]
        for d in self.ds:
            name, path = d['Experiment name'], d['Dataset path']
            if ('Finalization file' in d) and d['Finalization file'] and d['Finalization file'].startswith('exp'):
                fin_lbl = d['Finalization file']
                if os.path.exists(fn := os.path.join(path, f'{fin_lbl}.rrpprof')):
                    exps[name] = (path, fin_lbl)
                else:
                    raise FileNotFoundError(f'Specified profile file {fn} does not exist.')
            else:           
                for appendix in appendices:
                    if os.path.exists(os.path.join(path, f'{name}{appendix}.rrpprof')):
                        exps[name] = (path, f'{name}{appendix}')
                        if ('Finalization file' in d) and d['Finalization file']:
                            print(f'Warning: selected finalization {d["Finalization file"]} rejected as it seems to be a merged one. Using {exps[name]} instead.')                             
                        break
                else:
                    raise FileNotFoundError(f'No rrpprof file found for {name} in {path}.')                    
                            
        # create mangle-able ID strings from merging tree
        out_codes, in_codes = self.get_merge_codes()
        
        if short_form:
            out_paths = [os.path.join(exps[sorted(oc.split(':'))[0]][0] if common_path is None else common_path, 
                                  '-'.join([prefix, f'ID{ii+1:03d}', f'{len(oc.split(":"))}exp'])) 
                    for ii, oc in enumerate(out_codes)]
        else:        
            out_paths = [os.path.join(exps[sorted(fn.split(':'))[0]][0] if common_path is None else common_path, 
                                  '-'.join([prefix, fn]).replace(':', '-')) 
                    for fn in out_codes]
        
        # get info about which initial proffit files are in each output file
        # this corresponds to the out_code string ID, but with naked exp filenames replaced by the 
        # finalization name (to be fully unique) and sorted identifiers
        out_info = [
            ':'.join([exps[in_code][1] for in_code in sorted(out_code.split(':'))])
                    for out_code in out_codes]

        # get input paths
        if legacy:
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
                
        else:
            in_paths = []
            for prof_paths in out_codes:
                in_paths.append(
                    tuple(os.path.join(*exps[name]) for name in prof_paths.split(':'))
                    )
            
        return out_paths, in_paths, out_codes, out_info