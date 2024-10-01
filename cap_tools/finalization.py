from collections.abc import MutableMapping
import pandas as pd
import os
from io import StringIO
from typing import *
import xml.etree.ElementTree as ET
import warnings
import time


# try:
#     import gemmi
#     HAVE_GEMMI = True
# except ImportError:
#     HAVE_GEMMI = False   

HAVE_GEMMI = False # gemmi currently not required

_FOM_DICT = {'Rint': 1,
            'Rurim': 2,
            'Rpim': 3,
            'Sigma': 4,
            'SigmaA': 5,
            'SigmaB': 6,
            'CC 1/2': 7,
            'CC*': 8,
            'deltaCC': 9,
            'Rsym': 11,
            'RshelX': 12}


def write_fin_xml(template: str, name: str, folder: str, 
                  gral: Optional[bool] = None, autochem: Optional[bool] = None,
                  laue: Optional[Union[int]] = None, z: Optional[float] = None,
                  chem: Optional[str] = None, res_limit: Optional[float] = None,
                  fom: Union[list, tuple] = ('Rint', 'Rurim', 'Rpim', 'CC 1/2', 'deltaCC', 'Sigma', 'SigmaA', 'SigmaB', 'CC*'),
                  pars: Optional[Dict[str, str]] = None):

    tree = ET.parse(template)
    root = tree.getroot()
    root.find('__FINALIZER_SAMPLE__/__Input_file__').text = name
    root.find('__FINALIZER_SAMPLE__/__Input_file_path__').text = folder
    root.find('__FINALIZER_OUTPUT__/__Output_file__').text = name
    root.find('__FINALIZER_OUTPUT__/__Output_file_path__').text = os.path.join(folder, name)

    pars = {} if pars is None else pars
    
    if gral is not None:
        pars['__FINALIZER_SPACE_GROUP_AND_AUTOCHEM__/__Is_GRAL_on__'] = '1' if gral else '0'
    if autochem is not None:
        pars['__FINALIZER_SPACE_GROUP_AND_AUTOCHEM__/__Is_AutoChem_active__'] = '1' if autochem else '0'
    if autochem is not None:
        pars['__FINALIZER_SPACE_GROUP_AND_AUTOCHEM__/__Is_AutoChem_active__'] = '1' if autochem else '0'
    if laue is not None:
        if isinstance(laue, str):
            lcls = root.find('__FINALIZER_SAMPLE__/__Type_of_Laue__indexinfo__').text.split(';')
            laue = {v.strip(): int(k) for k, v in (lcl.split('-', 1) for lcl in lcls)}[laue]
        pars['__FINALIZER_SAMPLE__/__Type_of_Laue__'] = str(laue)
    if res_limit is not None:
        pars['__FINALIZER_FILTERS_AND_LIMITS__/__Automated__'] = '0'
        pars['__FINALIZER_FILTERS_AND_LIMITS__/__Apply_resolution_limits__'] = '1'
        pars['__FINALIZER_FILTERS_AND_LIMITS__/__Resolution_limits_-_high_limit__'] = str(res_limit)
        pars['__FINALIZER_FILTERS_AND_LIMITS__/__Dmin_for_completness__'] = str(res_limit)
    if z is not None:
        pars['__FINALIZER_SAMPLE__/__Z__'] = str(z)
    if chem is not None:
        pars['__FINALIZER_SAMPLE__/__Chemical_formula__'] = str(chem)
        
    pars['__FINALIZER_FILTERS_AND_LIMITS__/__Apply_printout_options__'] = '1'
    
    for ii, the_fom in enumerate(fom):
        # print(the_fom)
        pars[f'__FINALIZER_FILTERS_AND_LIMITS__/__Printout_options_-_Output_order_-_{ii}__'] = str(_FOM_DICT.get(the_fom, 0))
    
    # global settings
    for k, v in pars.items():
        try:
            root.find(k).text = v
        except AttributeError:
            print('Entry',k, 'not found in XML template.')
        
    xml_name = os.path.join(folder, name) + '_rrp.xml'
    tree.write(xml_name)
    
    return xml_name
   
   
class FinalizationXML:
    
    @classmethod
    def from_template(cls, template: str, path: str, filename: str):
        fin_xml = cls(template)
        fin_xml.filename = filename
        fin_xml.path = path
    
    def __init__(self, filename: str, path: str, allow_missing: bool = False, parse: bool = True):

        self.filename = filename
        self.path = path
        self.tree = None
        if parse:
            self.update(allow_missing)        
            
    def update(self, allow_missing: bool = False):
          
        try:
            self.tree = ET.parse(self.filename)      
        except FileNotFoundError as err:
            if allow_missing:
                self.tree = None
            else:
                raise FileNotFoundError(f'Finalization parameter file {self.filename} does not exist (yet).')
        
    def set_parameters(self, template: Optional[str] = None, 
                    gral: Optional[bool] = None, autochem: Optional[bool] = None,
                    laue: Optional[Union[int]] = None, z: Optional[float] = None,
                    chem: Optional[str] = None, res_limit: Optional[float] = None,
                    fom: Union[list, tuple] = ('Rint', 'Rurim', 'Rpim', 'CC 1/2', 'deltaCC', 'Sigma', 'SigmaA', 'SigmaB', 'CC*'),
                    pars: Optional[Dict[str, str]] = None):
        
        if template is not None:
            if os.path.exists(template):
                tree = ET.parse(template)
            else:
                raise FileNotFoundError(f'Finalization parameter template file {template} does not exist (yet).')
        elif self.tree is not None:
            tree = self.tree
        else:
            raise ValueError('No finalization parameters are loaded; please specify a template.')
            
        root = tree.getroot()
        root.find('__FINALIZER_SAMPLE__/__Input_file__').text = os.path.basename(self.path)
        root.find('__FINALIZER_SAMPLE__/__Input_file_path__').text = os.path.dirname(self.path)
        root.find('__FINALIZER_OUTPUT__/__Output_file__').text = os.path.basename(self.path)
        root.find('__FINALIZER_OUTPUT__/__Output_file_path__').text = self.path

        pars = {} if pars is None else pars
        
        if gral is not None:
            pars['__FINALIZER_SPACE_GROUP_AND_AUTOCHEM__/__Is_GRAL_on__'] = '1' if gral else '0'
        if autochem is not None:
            pars['__FINALIZER_SPACE_GROUP_AND_AUTOCHEM__/__Is_AutoChem_active__'] = '1' if autochem else '0'
        if autochem is not None:
            pars['__FINALIZER_SPACE_GROUP_AND_AUTOCHEM__/__Is_AutoChem_active__'] = '1' if autochem else '0'
        if laue is not None:
            if isinstance(laue, str):
                lcls = root.find('__FINALIZER_SAMPLE__/__Type_of_Laue__indexinfo__').text.split(';')
                laue = {v.strip(): int(k) for k, v in (lcl.split('-', 1) for lcl in lcls)}[laue]
            pars['__FINALIZER_SAMPLE__/__Type_of_Laue__'] = str(laue)
        if res_limit is not None:
            pars['__FINALIZER_FILTERS_AND_LIMITS__/__Automated__'] = '0'
            pars['__FINALIZER_FILTERS_AND_LIMITS__/__Apply_resolution_limits__'] = '1'
            pars['__FINALIZER_FILTERS_AND_LIMITS__/__Resolution_limits_-_high_limit__'] = str(res_limit)
            pars['__FINALIZER_FILTERS_AND_LIMITS__/__Dmin_for_completness__'] = str(res_limit)
        if z is not None:
            pars['__FINALIZER_SAMPLE__/__Z__'] = str(z)
        if chem is not None:
            pars['__FINALIZER_SAMPLE__/__Chemical_formula__'] = str(chem)
            
        pars['__FINALIZER_FILTERS_AND_LIMITS__/__Apply_printout_options__'] = '1'
        
        for ii, the_fom in enumerate(fom):
            # print(the_fom)
            pars[f'__FINALIZER_FILTERS_AND_LIMITS__/__Printout_options_-_Output_order_-_{ii}__'] = str(_FOM_DICT.get(the_fom, 0))
        
        # global settings
        for k, v in pars.items():
            try:
                root.find(k).text = v
            except AttributeError:
                print('Entry', k, 'not found in XML template.')
            
        self.tree = tree        
        
        if self.filename is not None:
            tree.write(self.filename)                   
        else:
            warnings.warn('No parameter XML file name set. Not writing changed parameters', RuntimeWarning)
   
class Finalization:
    """Extensible class to manage a CAP finalization run"""

    HEADLINE = 'Statistics vs resolution (taking redundancy into account)'

    def __init__(self, path: str, verbose: bool = True, 
                 meta: Optional[Dict] = None, sub_paths: Union[List[str], Tuple[str]] = (), 
                 allow_missing: bool = False, parse: bool = True):

        self.path: str = path
        self.verbose: bool = verbose
        self.shells: pd.DataFrame = pd.DataFrame([])
        self.overall: pd.DataFrame = pd.DataFrame([])
        self.sub_paths: List[str] = list(sub_paths)
        self.meta = meta if meta is not None else {}
        if (meta is not None) and ('Merge code' in meta):
            self.meta['Nexp'] = len(meta['Merge code'].split(':'))
        
        self.pars_xml = FinalizationXML(filename=self.pars_xml_path, 
                                        path=self.path, allow_missing=True, 
                                        parse=parse)
        
        if parse:
            
            if HAVE_GEMMI and os.path.exists(path + '.mtz'):
                self.mtz = gemmi.read_mtz_file(path + '.mtz')
                if verbose:
                    print(f'Parsed reflection file {path + ".mtz"}')
            else:
                self.mtz = None

                
            try:
                self.parse_finalization_results(check_current=True)       
            except FileNotFoundError as err:
                if not allow_missing:
                    raise err            
                elif verbose:
                    print(f'No result file found for {path}. Creating dummy finalization object') 
            except RuntimeError as err:
                warnings.warn(f'Result file for {path} is older than settings XML. Not parsing')
                    
    @property
    def foms(self):
        return list(self.shells.columns)
    
    @property
    def name(self):
        return os.path.basename(self.path)
    
    @property
    def folder(self):
        return os.path.dirname(self.path)
    
    @property
    def pars_xml_path(self):
        return self.path + '_rrp.xml'
    
    @property
    def have_proffit(self):
        return os.path.exists(self.path + '.rrpprof') 
    
    @property
    def have_pars_xml(self):
        return self.pars_xml.tree is not None
        
    def parse_finalization_parameters(self):
        
        fn = self.pars_xml_path
        
        if not os.path.exists(fn):
            raise FileNotFoundError(f'Result parameters file {fn} not found.')   
                
        self.pars_xml = ET.parse(fn)

    def parse_finalization_results(self, check_current: bool = True, timeout: float = 0):

        fn = self.path + '_red.sum'

        if not os.path.exists(fn):
            raise FileNotFoundError(f'Result summary file {fn} not found.')            
        
        if os.path.exists(self.pars_xml_path) and (os.path.getmtime(self.pars_xml_path) > os.path.getmtime(fn)):
            msg = f'Result summary {os.path.basename(fn)} is older than parameter file {os.path.basename(self.pars_xml_path)}'
            if check_current:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg, RuntimeWarning)

        if self.verbose:
            print(f'Parsing result summary file {fn}')            
        
        def get_table_section():
            table = ''
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
                    
            return table
        
        T = time.time() + timeout
        
        while (not (table := get_table_section())) and (time.time() < T):
            time.sleep(0.5)
        
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
                label_column: str = 'Name', 
                meta_cols: Union[List[str], Tuple[str]] = ('Cluster', 'Data sets', 'Merge code'),
                **kwargs):

        try:
            merge_sets = pd.read_csv(filename)
        except pd.errors.ParserError:
            merge_sets = pd.read_csv(filename, skiprows=7)
              
        fc = cls()
            
        for _, ds in merge_sets.iterrows():
            try:
                fc[ds[label_column]] = Finalization(ds['File path'], meta={k: ds[k] for k in meta_cols if k in ds}, 
                                                    **kwargs)
            except (RuntimeError, FileNotFoundError) as err:
                if ignore_parse_errors:
                    print(f'{ds["File path"]} not found or could not be parsed, skipping.')
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
    
    @classmethod
    def from_dict(cls, fins: dict):
        fc = cls()
        for k, v in fins.items():
            fc[k] = v
    
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
    
    def get_subset(self, names: Union[List, Tuple] = ()) -> 'FinalizationCollection':
        sub = FinalizationCollection()
        for n in names:
            try:
                sub[n] = self[n]
            except KeyError:
                raise KeyError(f'Finalization {n} not found.')
        return sub
    
    def sort_by_meta(self, by: Union[str, List[str]] = 'File path') -> 'FinalizationCollection':
        sort_list = list(self.meta.sort_values(by=by)['name'])
        return self.get_subset(sort_list)

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
    def meta(self) -> pd.DataFrame:
        return pd.DataFrame({name: {'File path': fin.path, **fin.meta} for name, fin  in self.items()}).T.reset_index(names='name')

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
        return {lbl: fin.path for lbl, fin in self.items()}
