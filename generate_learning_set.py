from glob import glob
import os
import pandas as pd
import shutil
import xml.etree.ElementTree as ET
import hashlib
from sys import argv
from argparse import ArgumentParser
from cap_tools.cap_control import CAPInstance, CAPListenModeError

parser = ArgumentParser(description='Generate anonymous ML training sets from CrysAlisPro ED datasets')
parser.add_argument('exp_dir')
parser.add_argument('out_dir')
parser.add_argument('--include-path', action='store_true')
args = parser.parse_args()

exp_dir = args.exp_dir
out_dir = args.out_dir
include_path = args.include_path

if not os.path.exists(exp_dir):
    raise FileNotFoundError(f'Folder {exp_dir} does not exist')

if include_path:
    print('WARNING: experiment path will be included in output list file. Resulting data will not be anonymized.')

print('Searching experiments in', exp_dir)
exp_list = [os.path.splitext(fn)[0] for fn in glob(os.path.join(exp_dir,'**\\*.par'), recursive=True) 
            if (('_cracker' not in fn) and ('tomo' not in fn) 
                and ('DD_Calib' not in fn) and ('Preset' not in fn) and ('Cluster' not in fn))]
root_dir = ''

print(len(exp_list), 'experiments of correct type found.')

root_dir = ''
os.makedirs(out_dir, exist_ok=True)

info = []
cap_cmds = []

for ii, exp in enumerate(sorted(exp_list)):
    info_fn = os.path.join(root_dir, os.path.dirname(exp), 'experiment_results.xmlinfo')
    info_str = open(info_fn).read()
    tree = ET.fromstring('<root>\n' + info_str + '\n</root>')
    exp_info = {'path': exp} if include_path else {}
    
    # generate hash digest stable information (not changing with reprocessing or moving)
    if (user := tree.find('__EXPERIMENT_INFO__/__USER__')) is not None:
        user = user.text
    else:
        user = 'anonymous'
        # print('WARNING: No user found for', exp)
        
    if (exp_time := tree.find('__EXPERIMENT_INFO__/__START_TIME__')) is not None:
        exp_time = exp_time.text
    else:
        exp_time = 'unknown time'
        # print('WARNING: No experiment time found for', exp)
        
    if (exp_name := tree.find('__EXPERIMENT_INFO__/__EXPERIMENT_PAR_NAME_WOEXT__')) is not None:
        exp_name = exp_name.text
    else:
        exp_name = os.path.basename(exp)
        # print('WARNING: No experiment name found for', exp)
    
    m = hashlib.md5()
    hash_text = ';'.join([user, exp_time, exp_name])
    m.update(hash_text.encode()) # this line defines what gets hashed
    basename = exp_info['digest'] = m.hexdigest()
    
    if ii == 0:
        # first loop run
        print('Generating anonymous experiment hashes from string of type:')
        print(hash_text)
        print('-->  ', basename)
    
    xml_entries = {
        'scan_range': tree.find('__EXPERIMENT_INFO__/__SCAN_RANGE__'),
        'detector_distance': tree.find('__EXPERIMENT_INFO__/__DETECTOR_DISTANCE__'),
        'indexation': tree.find('__EXPERIMENT_RESULTS__/__INDEXATION__'),
        'e1': tree.find('__EXPERIMENT_RESULTS__/__MOSAICITY__/__MOSAICITY_E1__'),
        'e2': tree.find('__EXPERIMENT_RESULTS__/__MOSAICITY__/__MOSAICITY_E2__'),
        'e3': tree.find('__EXPERIMENT_RESULTS__/__MOSAICITY__/__MOSAICITY_E3__'),
        'diff_limit': tree.find('__EXPERIMENT_RESULTS__/__DIFFLIMIT__'),
        'r_int': tree.find('__EXPERIMENT_RESULTS__/__RINT__'),
    }
    
    for k, v in xml_entries.items():
        if v is not None:
            exp_info[k] = float(v.text)
            
    if 'scan_range' not in exp_info:
        # print('No scan range found for', exp)
        continue

    fn_in = os.path.join(root_dir, exp) + '_middle_microed_diff_snapshot'
    fn_out = os.path.join(out_dir, basename) + '_diff.tiff'
    
    exp_info['diff_img'] = basename + '_diff.tiff'
    
    if os.path.exists(fn_out):
        pass
    elif os.path.exists(fn_in + '.tiff'):
        shutil.copy(fn_in + '.tiff', fn_out)
    elif os.path.exists(fn_in + '.rodhypix'):
        cap_cmds.append(f'rd i "{fn_in}.rodhypix"')
        cap_cmds.append(f'wd tiffopt {fn_out} 1 0 0 0')
    else:
        # print('No diffraction snapshot found for', exp)
        continue
    
        
    fn_in = os.path.join(root_dir, exp) + '_microed_grain_snapshot'
    fn_out = os.path.join(out_dir, basename) + '_grain.tiff'
    
    exp_info['grain_img'] = basename + '_grain.tiff'

    if os.path.exists(fn_out):
        pass              
    elif os.path.exists(fn_in + '.tiff'):
        shutil.copy(fn_in + '.tiff', fn_out)
    elif os.path.exists(fn_in + '.rodhypix'):
        cap_cmds.append(f'rd i {fn_in}.rodhypix')
        cap_cmds.append(f'wd tiffopt {fn_out} 1 0 0 0')
    else:
        # print('No grain snapshot found for', exp)
        continue
            
    info.append(exp_info)
    
info = pd.DataFrame(info)

print(len(info), 'experiments with sufficient meta data found.')

if os.path.exists(fn := os.path.join(out_dir, 'info.csv')):
    existing = pd.read_csv(fn)
    info = pd.concat([existing, info]).drop_duplicates(subset='digest')
    print(f'Existing dataset in {fn} had {len(existing)} entries. Now it has {len(info)} entries.')
   
info.to_csv(os.path.join(out_dir, 'info.csv'), index=False)

if cap_cmds: 
    # with open(os.path.join(out_dir, 'convert.mac'), 'w') as fh:
    #     fh.write('\n'.join(cap_cmds))
    # print(len(cap_cmds)//2, ' image/diffraction file conversions required. Please run this command in CrysAlisPro:')
    # print('SCRIPT', os.path.join(out_dir, 'convert.mac'))
    listen = CAPInstance()
    while True:
        try:
            listen.run_cmd(cap_cmds, use_mac=True)
            break
        except CAPListenModeError as err:
            print('-----')
            print(str(err))
            print('Press Return to Retry or Ctrl-C to quit.')
            try:
                input()
            except KeyboardInterrupt:
                print('Exiting.')
                exit()




print('Finished writing training data to:', out_dir)