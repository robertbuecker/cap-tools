from collections.abc import MutableMapping
from configparser import ConfigParser, Error as ConfigParserError
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path
import re
from typing import Any, Callable
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


class FinalizationLoadMode(str, Enum):
    CURRENT = "current"
    ALL = "all"
    PATTERNS = "patterns"


@dataclass(frozen=True)
class FinalizationLoadOptions:
    mode: FinalizationLoadMode = FinalizationLoadMode.CURRENT
    include_patterns: Tuple[str, ...] = ()
    exclude_patterns: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", FinalizationLoadMode(self.mode))
        object.__setattr__(self, "include_patterns", tuple(p for p in self.include_patterns if p))
        object.__setattr__(self, "exclude_patterns", tuple(p for p in self.exclude_patterns if p))

    def matches(self, name: str) -> bool:
        """Return whether a finalization basename passes pattern-mode rules."""
        lowered = name.casefold()
        included = any(fnmatchcase(lowered, pattern.casefold()) for pattern in self.include_patterns)
        excluded = any(fnmatchcase(lowered, pattern.casefold()) for pattern in self.exclude_patterns)
        if included:
            return True
        if self.include_patterns:
            return False
        return not excluded


@dataclass(frozen=True)
class Olex2Refinement:
    source: Optional[str] = None
    r1_gt: Optional[float] = None
    wr_ref: Optional[float] = None
    goof: Optional[float] = None

    @property
    def status(self) -> str:
        if self.source is None:
            return "not found"
        if all(value is not None for value in (self.r1_gt, self.wr_ref, self.goof)):
            return "complete"
        if any(value is not None for value in (self.r1_gt, self.wr_ref, self.goof)):
            return "partial"
        return "unrefined"

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {"R1_gt": self.r1_gt, "wR_ref": self.wr_ref, "GOOF": self.goof}


@dataclass(frozen=True)
class MergedExperiment:
    index: int
    name: str
    folder: Optional[str] = None
    rrpprof_name: Optional[str] = None


@dataclass(frozen=True)
class MergeMembership:
    inputs: Tuple[MergedExperiment, ...] = ()
    selected_indices: Optional[frozenset[int]] = None
    warnings: Tuple[str, ...] = ()

    @property
    def total_count(self) -> int:
        return len(self.inputs) if self.inputs else 1

    @property
    def used_count(self) -> Optional[int]:
        if not self.inputs:
            return 1
        if self.selected_indices is None:
            return None
        return sum(exp.index in self.selected_indices for exp in self.inputs)

    @property
    def used_names(self) -> Tuple[str, ...]:
        if not self.inputs or self.selected_indices is None:
            return ()
        return tuple(exp.name for exp in self.inputs if exp.index in self.selected_indices)

    @property
    def excluded_names(self) -> Tuple[str, ...]:
        if not self.inputs or self.selected_indices is None:
            return ()
        return tuple(exp.name for exp in self.inputs if exp.index not in self.selected_indices)


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def parse_olex2_refinement(finalization_path: str) -> Olex2Refinement:
    """Parse refinement metrics from the exact root-level Olex2 result file."""
    base = Path(finalization_path)
    source = base.parent / "struct" / f"olex2_{base.name}" / f"{base.name}.res"
    if not source.is_file():
        return Olex2Refinement()
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return Olex2Refinement(source=str(source))

    values: Dict[str, Optional[float]] = {"R1_gt": None, "wR_ref": None, "GOOF": None}
    for key in values:
        matches = re.findall(rf"^\s*REM\s+{re.escape(key)}\s*=\s*({_FLOAT_RE})\s*$", text, re.MULTILINE)
        if matches:
            try:
                values[key] = float(matches[-1])
            except ValueError:
                pass
    return Olex2Refinement(str(source), values["R1_gt"], values["wR_ref"], values["GOOF"])


def parse_merge_membership(finalization_path: str) -> MergeMembership:
    """Read merged experiment membership from the finalization's expinfo folder."""
    expinfo = Path(finalization_path).parent / "expinfo"
    merged_path = expinfo / "merged.ini"
    if not merged_path.is_file():
        return MergeMembership()

    parser = ConfigParser(interpolation=None)
    warnings_found: List[str] = []
    try:
        parser.read(merged_path, encoding="utf-8")
    except (OSError, UnicodeError, ConfigParserError) as err:
        return MergeMembership(warnings=(f"Could not parse merged.ini: {err}",))
    inputs: List[MergedExperiment] = []
    for section in parser.sections():
        match = re.fullmatch(r"Merged experiment\s+(\d+)", section, re.IGNORECASE)
        if not match:
            continue
        index = int(match.group(1))
        values = parser[section]
        name = values.get("experiment name", "").strip().strip('"')
        if not name:
            warnings_found.append(f"Missing experiment name for merged batch {index}")
            name = f"batch {index}"
        inputs.append(MergedExperiment(
            index=index,
            name=name,
            folder=values.get("experiment folder path", fallback=None),
            rrpprof_name=values.get("rrpprof name", fallback=None),
        ))
    inputs.sort(key=lambda value: value.index)

    try:
        declared = parser.getint(
            "Number of merged experiments", "number of merged experiments", fallback=len(inputs)
        )
    except ValueError:
        declared = len(inputs)
        warnings_found.append("Invalid input count in merged.ini")
    if declared != len(inputs):
        warnings_found.append(f"merged.ini declares {declared} inputs but lists {len(inputs)}")

    clustering_path = expinfo / "clustering.ini"
    if not clustering_path.is_file():
        return MergeMembership(tuple(inputs), None, tuple(warnings_found))

    clustering = ConfigParser(interpolation=None)
    try:
        clustering.read(clustering_path, encoding="utf-8")
    except (OSError, UnicodeError, ConfigParserError) as err:
        warnings_found.append(f"Could not parse clustering.ini: {err}")
        return MergeMembership(tuple(inputs), None, tuple(warnings_found))
    selected: set[int] = set()
    seen: set[int] = set()
    if clustering.has_section("Selected batches"):
        for key, value in clustering.items("Selected batches"):
            match = re.fullmatch(r"is\s+(\d+)\s+batch\s+selected", key, re.IGNORECASE)
            if not match:
                continue
            index = int(match.group(1))
            seen.add(index)
            if value.strip() == "1":
                selected.add(index)
            elif value.strip() != "0":
                warnings_found.append(f"Invalid selection flag for merged batch {index}: {value}")
    expected = {exp.index for exp in inputs}
    if seen != expected:
        warnings_found.append("clustering.ini selection indices do not match merged.ini")
        return MergeMembership(tuple(inputs), None, tuple(warnings_found))
    return MergeMembership(tuple(inputs), frozenset(selected), tuple(warnings_found))

class FinalizationXML:
    # TODO Change this to a pure _parser_ without write functionality. It should only extract a few key parameters from the finalization XML.
    # TODO For now, it's fairly useless.
    
    @classmethod
    def from_template(cls, template_file: str, path: str, filename: str):
        """Generate an XML parameter set from a template XML file, but setting a new filename

        Args:
            template_file (str): Template XML finalization file generated from CAP
            path (str): Name of finalization (including full experiment path, without extension)
            filename (str): File name of new XML file
        """
        fin_xml = cls(template_file)
        fin_xml.filename = filename
        fin_xml.path = path
        #TODO there seem to be two template mechanisms? This never seems to be called.
    
    def __init__(self, filename: str, path: str, allow_missing: bool = False, parse: bool = True):
        """Open a CAP finalization XML file as generated by `DC RRP` or `DC XMLRRP`

        Args:
            filename (str): XML finalization file generated by CAP
            path (str): Name of finalization (including full experiment path, without extension)
            allow_missing (bool, optional): Allows to create an instance without an actual file. Defaults to False.
            parse (bool, optional): Parse the XML file straight away. Otherwise, it can be done later using `FinalizationXML.update`. 
            Defaults to True.
        """        

        self.filename = filename
        self.path = path
        self.tree = None
        if parse:
            self.update(allow_missing)        
            
    def update(self, allow_missing: bool = False):
        """Updates CAP finalization settings from the XML file

        Args:
            allow_missing (bool, optional): If True and the file is missing, set the internal tree to None instead of
            raising an error. Defaults to False.

        Raises:
            FileNotFoundError: _description_
        """
          
        try:
            self.tree = ET.parse(self.filename)      
        except FileNotFoundError as err:
            if allow_missing:
                self.tree = None
            else:
                raise FileNotFoundError(f'Finalization parameter file {self.filename} does not exist (yet).')
        
    def set_parameters(self, template: Optional[str] = None, 
                    gral: Optional[bool] = None, gral_interactive: Optional[bool] = None,
                    autochem: Optional[bool] = None,
                    laue: Optional[Union[int]] = None, z: Optional[float] = None,
                    chem: Optional[str] = None, res_limit: Optional[float] = None,
                    fom: Union[list, tuple] = ('Rint', 'Rurim', 'Rpim', 'CC 1/2', 'deltaCC', 'Sigma', 'SigmaA', 'SigmaB', 'CC*'),
                    N_shells: int = 10,
                    pars: Optional[Dict[str, str]] = None):
        #TODO why is there another template mechanism here?
        #TODO why does the _parser_ need a function to overwrite the template? Isn't that covered by `cap-auto` now?
        
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
        if gral_interactive is not None:
            pars['__FINALIZER_SPACE_GROUP_AND_AUTOCHEM__/__GRAL_mode__'] = '1' if gral_interactive else '0'            
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
        pars['__FINALIZER_FILTERS_AND_LIMITS__/__Printout_options_-_number_of_shells__'] = str(N_shells)
        
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

    def __init__(self, path: str, verbose: bool = True, merged: bool = False,
                 meta: Optional[Dict] = None, sub_paths: Union[List[str], Tuple[str]] = (), 
                 allow_missing: bool = False, parse: bool = True):
        # TODO: document this properly. OMG.

        self.path: str = path
        self.verbose: bool = verbose
        self.shells: pd.DataFrame = pd.DataFrame([])
        self.overall: pd.DataFrame = pd.DataFrame([])
        self.merged: bool = merged
        self.sub_paths: List[str] = list(sub_paths)
        self.meta = meta if meta is not None else {}
        if (meta is not None) and ('Merge code' in meta):
            self.meta['Nexp'] = len(meta['Merge code'].split(':'))
        self.refinement = parse_olex2_refinement(path)
        self.merge_membership = parse_merge_membership(path)
        
        # skipping the XML parsing. It's not mandatory as the code does not actually run the finalizations anymore.
        # self.pars_xml = FinalizationXML(filename=self.pars_xml_path, 
        #                                path=self.path, allow_missing=True, 
        #                                parse=parse)
        
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
        return self.path + ('_finalizer.xml' if not self.merged else '_finalizer_default_merged.xml') 
    
    @property
    def have_proffit(self):
        return os.path.exists(self.path + '.rrpprof') 

    @property
    def olex2_res_path(self) -> Optional[str]:
        return self.refinement.source

    @property
    def olex2_metrics(self) -> Dict[str, Optional[float]]:
        return self.refinement.as_dict()

    @property
    def merged_metadata(self) -> Dict[str, Any]:
        membership = self.merge_membership
        return {
            'N inputs': membership.total_count,
            'N used': membership.used_count,
            'Input experiments': membership.inputs and ', '.join(exp.name for exp in membership.inputs) or '',
            'Used experiments': ', '.join(membership.used_names),
            'Excluded experiments': ', '.join(membership.excluded_names),
        }
    
    @property
    def have_pars_xml(self):
        return False
        # XML parsing is temporarily disabled. The XML file is not required for the finalization viewer, and the parsing is currently broken.
        # return self.pars_xml.tree is not None

    def parse_finalization_results(self, check_current: bool = False, timeout: float = 0):

        fn = self.path + '_red.sum'

        if not os.path.exists(fn):
            raise FileNotFoundError(f'Result summary file {fn} not found.')            
        
        # We don't check the timestamp of the XML file anymore, as it is not required for the finalization viewer. The XML parsing is currently disabled.
        # if os.path.exists(self.pars_xml_path) and (os.path.getmtime(self.pars_xml_path) > (os.path.getmtime(fn) + 5)):
        #     msg = f'Result summary {os.path.basename(fn)} is older than parameter file {os.path.basename(self.pars_xml_path)}'
        #     if check_current:
        #         raise RuntimeError(msg)
        #     else:
        #         warnings.warn(msg, RuntimeWarning)

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
        ov_high = pd.concat((self.overall.iloc[[0],:], self.highest_shell)).reset_index(drop=True).drop(columns='1/d').astype(str).transpose()
        return pd.DataFrame('' + ov_high[0] + ' (' + ov_high[1] + ')').transpose()    
    

class FinalizationCollection(MutableMapping[str, Finalization]):
    """Manages are collection of finalizations with a dict-like interface and nice auto-functions"""

    @classmethod
    def from_folder(cls, folder: str, include_subfolders: bool = False,
                    ignore_parse_errors: bool = False,
                    load_options: Optional[FinalizationLoadOptions] = None,
                    **kwargs):
        folder_path = Path(folder)
        options = load_options or FinalizationLoadOptions()
        iterator = folder_path.rglob('*_red.sum') if include_subfolders else folder_path.glob('*_red.sum')
        discovered = [
            path.with_name(path.name[:-8])
            for path in iterator
            if not (len(path.parts) >= 2 and tuple(part.casefold() for part in path.parts[-3:-1]) == ('struct', 'tmp'))
        ]
        candidates = discovered
        if options.mode is FinalizationLoadMode.PATTERNS:
            candidates = [path for path in candidates if options.matches(path.name)]

        fc = cls()
        fc.candidate_count = len(discovered)
        if options.mode is FinalizationLoadMode.CURRENT:
            grouped: Dict[Path, List[Path]] = {}
            for path in candidates:
                grouped.setdefault(path.parent, []).append(path)
            for parent in sorted(grouped, key=lambda value: str(value).casefold()):
                ordered = sorted(
                    grouped[parent],
                    key=lambda value: value.with_name(value.name + '_red.sum').stat().st_mtime,
                    reverse=True,
                )
                failures: List[str] = []
                for path in ordered:
                    try:
                        fc._add_finalization(Finalization(str(path), **kwargs))
                        fc._loaded_candidate_count += 1
                        if failures:
                            fc.load_messages.append(
                                f"Used {path.name} after skipping newer malformed finalization(s): "
                                + '; '.join(failures)
                            )
                        break
                    except (OSError, RuntimeError, ValueError, KeyError, IndexError, UnicodeError, pd.errors.ParserError) as err:
                        failures.append(f"{path.name}: {err}")
                        fc.malformed_count += 1
                else:
                    if failures:
                        fc.load_messages.append(f"No parseable finalization in {parent}: {'; '.join(failures)}")
        else:
            for path in sorted(candidates, key=lambda value: str(value).casefold()):
                fc._try_add(str(path), ignore_parse_errors=ignore_parse_errors, **kwargs)
        fc._finalize_diagnostics()
        fc._emit_load_warnings()
        return fc
    
    @classmethod
    def from_csv(cls, filename: str,
                ignore_parse_errors: bool = False, 
                label_column: str = 'Experiment_name', 
                meta_cols: Union[List[str], Tuple[str]] = ('Cluster', 'Data sets', 'Merge code'),
                load_options: Optional[FinalizationLoadOptions] = None,
                **kwargs):
        options = load_options or FinalizationLoadOptions()
        try:
            merge_sets = pd.read_csv(filename)
        except (pd.errors.ParserError, UnicodeDecodeError):
            merge_sets = pd.read_csv(filename, skiprows=7)

        if 'Experiment_path' not in merge_sets.columns:
            merge_sets = pd.read_csv(filename, skiprows=7)

        fc = cls()
        for _, ds in merge_sets.iterrows():
            experiment_path = Path(str(ds.get('Experiment_path', '')))
            meta = {k: ds[k] for k in meta_cols if k in ds and pd.notna(ds[k])}
            if label_column in ds and pd.notna(ds[label_column]):
                meta['Experiment'] = ds[label_column]
            candidates: List[Path]
            if options.mode is FinalizationLoadMode.CURRENT:
                fc.candidate_count += 1
                selected = ds.get('Finalization_output_file')
                if pd.isna(selected) or str(selected).strip() in {'', '---'}:
                    fc.load_messages.append(f"No current finalization recorded for {experiment_path}")
                    continue
                candidates = [experiment_path / str(selected)]
            else:
                discovered = [path.with_name(path.name[:-8]) for path in experiment_path.glob('*_red.sum')]
                fc.candidate_count += len(discovered)
                candidates = discovered
                if options.mode is FinalizationLoadMode.PATTERNS:
                    candidates = [path for path in candidates if options.matches(path.name)]
            for path in sorted(candidates, key=lambda value: str(value).casefold()):
                fc._try_add(str(path), meta=meta.copy(), ignore_parse_errors=ignore_parse_errors, **kwargs)
        fc._finalize_diagnostics()
        fc._emit_load_warnings()
        return fc
    
    @classmethod
    def from_files(cls, filenames: List[str], **kwargs):
        fc = cls()
        for path in filenames:
            fc.candidate_count += 1
            fc._try_add(os.path.normpath(path), ignore_parse_errors=False, **kwargs)
        return fc
    
    @classmethod
    def from_dict(cls, fins: dict):
        fc = cls()
        for k, v in fins.items():
            fc[k] = v
        return fc
    
    def __init__(self):
        super().__init__()
        self._finalizations: Dict[str, Finalization] = {}
        self.load_messages: List[str] = []
        self.candidate_count = 0
        self.skipped_count = 0
        self.malformed_count = 0
        self._loaded_candidate_count = 0

    def _try_add(self, path: str, *, ignore_parse_errors: bool, **kwargs) -> bool:
        try:
            self._add_finalization(Finalization(path, **kwargs))
            self._loaded_candidate_count += 1
            return True
        except (OSError, RuntimeError, ValueError, KeyError, IndexError, UnicodeError, pd.errors.ParserError) as err:
            self.malformed_count += 1
            message = f'{path} could not be parsed, skipping. Error was: {err}'
            self.load_messages.append(message)
            if not ignore_parse_errors:
                raise
            return False

    def _finalize_diagnostics(self) -> None:
        self.skipped_count = max(0, self.candidate_count - self._loaded_candidate_count)

    def _add_finalization(self, finalization: Finalization) -> str:
        for key, existing in self.items():
            if os.path.normcase(os.path.abspath(existing.path)) == os.path.normcase(os.path.abspath(finalization.path)):
                return key
        base = finalization.name
        key = base
        if key in self._finalizations:
            parent = Path(finalization.path).parent
            depth = 1
            while key in self._finalizations:
                qualifier = os.path.join(*parent.parts[-depth:]) if depth <= len(parent.parts) else str(parent)
                key = f'{base} [{qualifier}]'
                depth += 1
                if depth > len(parent.parts) + 1 and key in self._finalizations:
                    key = f'{base} [{len(self._finalizations) + 1}]'
                    break
        self[key] = finalization
        return key

    def _emit_load_warnings(self) -> None:
        for message in self.load_messages:
            warnings.warn(message, RuntimeWarning, stacklevel=2)

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

        return pd.concat(summary, join='outer') if summary else pd.DataFrame(columns=['name'])

    @property
    def shelldata(self) -> pd.DataFrame:
        allshell = []
        for k, v in self.items():
            shells = v.shells.copy()
            shells['name'] = k
            allshell.append(shells)

        return pd.concat(allshell, join='outer') if allshell else pd.DataFrame(columns=['name'])

    @property
    def highest_shell(self) -> pd.DataFrame:
        allshell = []
        for k, v in self.items():
            shells = v.highest_shell.copy()
            shells['name'] = k
            allshell.append(shells)
    
        return pd.concat(allshell, join='outer') if allshell else pd.DataFrame(columns=['name'])

    @property
    def overall_highest(self) -> pd.DataFrame:
        allshell = []
        for k, v in self.items():
            shells = v.overall_highest.copy()
            shells['name'] = k
            allshell.append(shells)
     
        return pd.concat(allshell, join='outer') if allshell else pd.DataFrame(columns=['name'])
    
    @property
    def meta(self) -> pd.DataFrame:
        rows = {}
        for name, fin in self.items():
            rows[name] = {
                'File path': fin.path,
                **fin.meta,
                **fin.merged_metadata,
                **fin.olex2_metrics,
                'Olex2 status': fin.refinement.status,
            }
        return pd.DataFrame(rows).T.reset_index(names='name')

    @property
    def overall_numeric(self) -> pd.DataFrame:
        """Numeric overall statistics combined with refinement and merge metadata."""
        overall = self.overall.copy()
        if overall.empty:
            return pd.DataFrame(columns=['name'])
        return overall.merge(self.meta, on='name', how='left')

    @property
    def highest_numeric(self) -> pd.DataFrame:
        highest = self.highest_shell.copy()
        if highest.empty:
            return pd.DataFrame(columns=['name'])
        return highest.merge(self.meta, on='name', how='left')

    @property
    def shell_table(self) -> pd.DataFrame:
        return self.shelldata.pivot(columns='name', index='dmin').sort_index(ascending=False)
    
    def get_shell_table(self, fom=Union[str, List[str]]) -> pd.DataFrame:
        return self.shelldata.pivot(columns='name', index='dmin', values=fom).sort_index(ascending=False)
    
    @property
    def foms(self):
        seen = set()
        return [f for fin in self.values() for f in fin.foms if not (f in seen or seen.add(f))]
    
    @property
    def path(self):
        return {lbl: fin.path for lbl, fin in self.items()}


