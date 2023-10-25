import numpy as np
from scipy.cluster.hierarchy import linkage
import yaml
from cap_tools.interact_figures import distance_from_dendrogram, find_cell
from cap_tools.utils import get_clusters, parse_cap_csv, put_in_order, to_radian, to_sin, unit_cell_lcv_distance, write_cap_csv
from typing import *
from cap_tools.utils import volume_difference


def cluster_cell(cells: list, 
                 distance: float=None, 
                 method: str="average", 
                 metric: str="euclidean", 
                 use_radian: bool=False,
                 use_sine: bool=False,
                 labels: Optional[List[str]] = None,
                 fig = None):
    """Perform hierarchical cluster analysis on a list of cells. 

    method: lcv, volume, euclidean
    distance: cutoff distance, if it is not given, pop up a dendrogram to
        interactively choose a cutoff distance
    use_radian: Use radian instead of degrees to downweight difference
    use_sine: Use sine for unit cell clustering (to disambiguousize the difference in angles)
    """

    from scipy.spatial.distance import pdist

    if use_sine:
        _cells = to_sin(cells)
    elif use_radian:
        _cells = to_radian(cells)
    else:
        _cells = cells

    if metric == "lcv":
        dist = pdist(_cells, metric=unit_cell_lcv_distance)
        z = linkage(dist,  method=method)
        initial_distance = None
    elif metric == "volume":
        dist = pdist(_cells, metric=volume_difference)
        z = linkage(dist,  method=method)
        initial_distance = 250.0
    else:
        z = linkage(_cells,  metric=metric, method=method)
        initial_distance = 2.0

    if not distance:
        distance = distance_from_dendrogram(z, ylabel=metric, initial_distance=initial_distance, labels=labels, fig_handle=fig)

    print(f"Linkage method = {method}")
    print(f"Cutoff distance = {distance}")
    print(f"Distance metric = {metric}")
    print("")

    return get_clusters(z, cells, distance=distance)


def main():
    import argparse

    description = "Program for finding the unit cell from a serial crystallography experiment."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        
    parser.add_argument("args",
                        type=str, nargs="*", metavar="FILE",
                        help="Path to .yaml (edtools) or .csv (CrysAlisPro) file")

    parser.add_argument("-b","--binsize",
                        action="store", type=float, dest="binsize",
                        help="Binsize for the histogram, default=0.5")

    parser.add_argument("-c","--cluster",
                        action="store_true", dest="cluster",
                        help="Apply cluster analysis instead of interactive cell finding")

    parser.add_argument("-d","--distance",
                        action="store", type=float, dest="distance",
                        help="Cutoff distance to use for clustering, bypass dendrogram")

    parser.add_argument("-m","--method",
                        action="store", type=str, dest="method",
                        choices="single average complete median weighted centroid ward".split(),
                        help="Linkage algorithm to use (see `scipy.cluster.hierarchy.linkage`)")

    parser.add_argument("-t","--metric",
                        action="store", type=str, dest="metric",
                        choices="euclidean lcv volume".split(),
                        help="Metric for calculating the distance between items (Euclidian distance, cell volume, LCV as in CCP4-BLEND)")

    parser.add_argument("-l", "--use_bravais_lattice",
                        action="store_false", dest="use_raw_cell",
                        help="Use the bravais lattice (symmetry applied)")

    parser.add_argument("-r", "--use_radian_for_angles",
                        action="store_true", dest="use_radian_for_clustering",
                        help="Use radians for unit cell clustering (to downweight the difference in angles)")

    parser.add_argument("-s", "--use_sine_for_angles",
                        action="store_true", dest="use_sine_for_clustering",
                        help="Use sine for unit cell clustering (to disambiguousize the difference in angles)")
    
    parser.add_argument("-w","--raw-cell",
                       action="store_true", dest="use_raw_cell",
                       help="Use the raw lattice (from Lattice Explorer/IDXREF as opposed to the refined one from GRAL/CORRECT) for unit cell finding and clustering")

    parser.set_defaults(binsize=0.5,
                        cluster=False,
                        distance=None,
                        method="average",
                        metric="euclidean",
                        use_raw_cell=False,
                        raw=False,
                        use_radian_for_clustering=False,
                        use_sine_for_clustering=False)
    
    options = parser.parse_args()

    distance = options.distance
    binsize = options.binsize
    cluster = options.cluster
    method = options.method
    metric = options.metric
    use_raw_cell = options.use_raw_cell
    use_radian = options.use_radian_for_clustering
    use_sine = options.use_sine_for_clustering
    args = options.args

    if args:
        fn = args[0]
    else:
        fn = "cells.yaml"
        fn = 'result-viewer.csv'
        
    if fn.endswith('.yaml') or fn.endswith('.yml'):
        use_yaml = True
        ds = yaml.load(open(fn, "r"), Loader=yaml.Loader)
        key = "raw_unit_cell" if use_raw_cell else "unit_cell"            
        # prune based on NaNs (missing cells)
        ds = [d for d in ds if not any(np.isnan(d[key]))]
        cells = np.array([d[key] for d in ds])
        weights = np.array([d["weight"] for d in ds])
        
    elif fn.endswith('.csv'):
        use_yaml = False
        ds, cells, weights = parse_cap_csv(fn, use_raw_cell)
        
    else:
        raise ValueError('Input file must be .yaml (edtools/XDS) or .csv (CrysAlisPro)')

    cells = put_in_order(cells)    

    if cluster:
        if not use_yaml:
            try:
                labels = [d['Experiment name'] for d in ds]
            except KeyError as err:
                print('Experiment names not found in CSV list. Consider including them.')
                labels = None
        else:
            labels = None
        clusters = cluster_cell(cells, distance=distance, method=method, metric=metric, use_radian=use_radian, use_sine=use_sine, labels=labels)
        for i, idx in clusters.items():
            clustered_ds = [ds[i] for i in idx]
            if use_yaml:
                fout = f"cells_cluster_{i}_{len(idx)}-items.yaml"
                yaml.dump(clustered_ds, open(fout, "w"))
            else:
                fout = f"{fn.rsplit('.', 1)[0]}_cells_cluster_{i}_{len(idx)}-items.csv"
                write_cap_csv(fout, clustered_ds)                                
                      
            print(f"Wrote cluster {i} to file `{fout}`")
    
    else:
        constants, esds = find_cell(cells, weights, binsize=binsize)
        
        print()
        print("Weighted mean of histogram analysis")
        print("---")
        print("Unit cell parameters: ", end="")
        for c in constants:
            print(f"{c:8.3f}", end="")
        print()
        print("Unit cell esds:       ", end="")
        for e in esds:
            print(f"{e:8.3f}", end="")
        print()

        try:
            import uncertainties as u
        except ImportError:
            pass
        else:
            print()
            names = (("a"," Å"), ("b"," Å"), ("c"," Å"),
                     ("α", "°"), ("β", "°"), ("γ", "°"))
            for i, (c, e) in enumerate(zip(constants, esds)):
                name, unit = names[i]
                val = u.ufloat(c, e)
                end = ", " if i < 5 else "\n"
                print(f"{name}={val:.2uS}{unit}", end=end)

        print()
        print("UNIT_CELL_CONSTANTS= " + " ".join(f"{val:.3f}" for val in constants))


if __name__ == '__main__':
    main()
