from collections.abc import MutableMapping
import pandas as pd
import os
from io import StringIO
from typing import *

try:
    import gemmi
    HAVE_GEMMI = True
except ImportError:
    HAVE_GEMMI = False   

class Finalization:
    """Extensible class to manage a CAP finalization run"""

    HEADLINE = 'Statistics vs resolution (taking redundancy into account)'

    def __init__(self, path: str, verbose: bool = True):

        self.basename = path
        self.verbose = verbose
        self.shells = pd.DataFrame([])
        self.overall = pd.DataFrame([])
        
        if HAVE_GEMMI and os.path.exists(path + '.mtz'):
            self.mtz = gemmi.read_mtz_file(path + '.mtz')
            if verbose:
                print(f'Parsed reflection file {path + ".mtz"}')
        else:
            self.mtz = None

        self.parse_finalization_results()

        
    @property
    def foms(self):
        return list(self.shells.columns)

    def parse_finalization_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        fn = self.basename + '_red.sum'

        table = ''

        if not os.path.exists(fn):
            raise RuntimeError(f'Result summary file {fn} not found.')            

        if self.verbose:
            print(f'Parsing result summary file {fn}')

        with open(fn, 'r') as fh:
            # slice out result table from summary file (do not parse values yet)
            parsing = False
            for ln in fh:
                # print(ln)
                if parsing and ((not ln.strip()) or (ln.strip().startswith('Data') or ln.strip().startswith('* * *'))):
                # if parsing and not ln.strip():
                    parsing = False
                    continue
                elif parsing:
                    table += ln
                elif ln.startswith(self.HEADLINE):
                    table = ''
                    parsing = True
                    continue

            if not table:
                raise RuntimeError(f'No result table found in {fn}')            
            else:
                if self.verbose:
                    print(f'Found result table in {fn}:')
                    print(table)

        sh, res, overall = StringIO(table), StringIO(), StringIO()
        _, cols, _ = [sh.readline() for _ in range(3)] # get header lines

        # parse shell and overall data
        parse_ov = False
        for ln in sh:
            if ln.startswith('-----------------------------'):
                parse_ov = True
                continue
            if parse_ov:
                overall.write(ln)
            else:
                res.write(ln)
        res.seek(0), overall.seek(0)

        # mangle column names
        cols = cols.replace('tion(A)', 'dmax dmin').replace('CC 1/2', 'CC1/2').split()

        # import final data into Pandas dataframes and mangle a bit
        shells = pd.read_csv(res, skiprows=0, header=None, sep=r'(?<=[^\s])-\s*|\s+', names=cols, engine='python')
        overall = pd.read_csv(overall, skiprows=0, header=None, sep=r'(?<=[^\s])-\s*|\s+', names=cols, engine='python')
        # shells['dmax'] = shells['dmax'].str.split('-',expand=True)[0].astype(float)
        # overall['dmax'] = overall['dmax'].str.split('-',expand=True)[0].astype(float)
        
        shells['1/d'] = (1/shells['dmax'] + 1/shells['dmin'])/2
        overall['1/d'] = (1/overall['dmax'] + 1/overall['dmin'])/2

        self.shells, self.overall = shells, overall

    @property
    def highest_shell(self) -> pd.DataFrame:
        return self.shells.iloc[[-1],:]
    
    @property
    def overall_highest(self) -> pd.DataFrame:
        ov_high = pd.concat((self.overall, self.highest_shell)).reset_index(drop=True).drop(columns='1/d').astype(str).transpose()
        return pd.DataFrame('' + ov_high[0] + ' (' + ov_high[1] + ')').transpose()    
    

class FinalizationCollection(MutableMapping[str, Finalization]):
    """Manages are collection of finalizations with a dict-like interface and nice auto-functions"""

    @classmethod
    def from_folder(cls, folder: str, include_subfolders: bool = False,
                    ignore_parse_errors: bool = False, **kwargs):
        from glob import glob
        if include_subfolders:
                paths = [fn[:-8] for fn in glob(os.path.join(folder, '*_red.sum'))]
            
        paths = [fn[:-8] for fn in glob(os.path.join(folder, '**', '*_red.sum'), recursive=True)]

        fc = cls()

        for path in paths:
            try:
                fc[os.path.basename(path)] = Finalization(path, **kwargs)
            except RuntimeError as err:
                if ignore_parse_errors:
                    print(f'{path} could not be parsed, skipping.')
                else:
                    raise err

        return fc
    
    @classmethod
    def from_csv(cls, filename: str,
                ignore_parse_errors: bool = False, 
                label_column='Name', **kwargs):
        
        import csv
        with open(filename) as fh:
            merge_sets = list(csv.DictReader(fh)) 
            
        fc = cls()
            
        for ds in merge_sets:
            try:
                fc[ds[label_column]] = Finalization(ds['File path'], **kwargs)
            except RuntimeError as err:
                if ignore_parse_errors:
                    print(f'{ds["File path"]} could not be parsed, skipping.')
                else:
                    raise err

        return fc
    
    @classmethod
    def from_files(cls, filenames: List[str], **kwargs):

        fc = cls()

        for path in filenames:
            try:
                fc[os.path.basename(path)] = Finalization(path, **kwargs)
            except RuntimeError as err:
                print(f'{path} could not be parsed, skipping.')
                raise err

        return fc
    
    def __init__(self):
        super().__init__()
        self._finalizations: Dict[str, Finalization] = {}

    def __setitem__(self, key: str, value: Finalization):
        self._finalizations[key] = value

    def __getitem__(self, key) -> Finalization:
        try:
            return self._finalizations[key]
        except KeyError as err:
            raise KeyError(f'Finalization {key} not found in collection')

    def __len__(self) -> int:
        return len(self._finalizations)

    def __delitem__(self, key: str):
        del self._finalizations[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._finalizations)

    @property
    def overall(self) -> pd.DataFrame:
        summary = []
        for k, v in self.items():
            ov = v.overall.iloc[[0],:].copy()
            ov['name'] = k
            summary.append(ov)

        return pd.concat(summary, join='outer')

    @property
    def shelldata(self) -> pd.DataFrame:
        allshell = []
        for k, v in self.items():
            shells = v.shells.copy()
            shells['name'] = k
            allshell.append(shells)

        return pd.concat(allshell, join='outer')

    @property
    def highest_shell(self) -> pd.DataFrame:
        allshell = []
        for k, v in self.items():
            shells = v.highest_shell.copy()
            shells['name'] = k
            allshell.append(shells)
    
        return pd.concat(allshell, join='outer')

    @property
    def overall_highest(self) -> pd.DataFrame:
        allshell = []
        for k, v in self.items():
            shells = v.overall_highest.copy()
            shells['name'] = k
            allshell.append(shells)
    
        return pd.concat(allshell, join='outer')

    @property
    def shell_table(self) -> pd.DataFrame:
        return self.shelldata.pivot(columns='name', index='dmin').sort_index(ascending=False)
    
    def get_shell_table(self, fom=Union[str, List[str]]) -> pd.DataFrame:
        return self.shelldata.pivot(columns='name', index='dmin', values=fom).sort_index(ascending=False)
    
    @property
    def foms(self):
        return list({f for fin in self.values() for f in fin.foms})
    
    @property
    def path(self):
        return {lbl: fin.basename for lbl, fin in self.items()}
