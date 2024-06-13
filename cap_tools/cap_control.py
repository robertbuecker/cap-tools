import os
from cap_tools.finalization import Finalization, FinalizationCollection
import glob
import time
import pandas as pd





def cluster_finalize(cluster_name: str,
                     include_proffitmerge: bool = True, res_limit: float = 0.8,
                     finalization_timeout: float = 10, 
                     _write_mac_only: bool = False, _skip_mac_write: bool = False):    


    cluster_result = cluster_name + '_merge_info.csv'
    merge_macro = cluster_name + '_merge.mac'
    xml_folder = os.path.join(os.path.dirname(cluster_result), 'finalization_templates')
    os.makedirs(xml_folder, exist_ok=True)

    fns = []
    for path in pd.read_csv(cluster_result)['File path']:
        fns += glob.glob(path + '*.*')
    fns += glob.glob(os.path.join(xml_folder, '*.xml'))
    time.sleep(0.1)
    for fn in fns:        
        os.remove(fn)
        print(f'Deleting {fn}')
        
    # %% load finalization info CSV and delete old files
    fc = FinalizationCollection.from_csv(cluster_result, allow_missing=True, verbose=False)

    # %% append commands to create dummy finalization XMLs from top nodes (which are never executed)

    if include_proffitmerge:
        with open(merge_macro) as fh:
            cmds = [ln.strip() for ln in fh]
    else:
        cmds = []

    top_node_names = list(fc.meta.sort_values(by=['Cluster', 'Nexp', 'File path']).drop_duplicates(subset='Cluster', keep='last')['name'])
    top_nodes = fc.get_subset(top_node_names)

    template_files = {}
    for name, fin in top_nodes.items():
        folder = os.path.dirname(fin.path)
        the_xml = os.path.join(xml_folder, f'C{fin.meta["Cluster"]}.xml')
        cmds.append(f'xx selectexpnogui ' + os.path.join(folder, os.path.split(folder)[-1] + ".par"))
        cmds.append(f'dc xmlrrp {name} ' + the_xml)
        template_files[fin.meta['Cluster']] = the_xml
        
    # append commands to run finalizations based on individual XML for each node (assuming they exist!)
    cmds.append('xx sleep 1000')
    prev_folder = ''

    for name, fin in fc.sort_by_meta(by=['Cluster', 'File path']).items():
        # print(node)
        folder = fin.folder
        
        if folder != prev_folder:
            par =  os.path.basename(folder)
            cmds.append(f'xx selectexpnogui {os.path.join(folder, par) + ".par"}')
            
        cmds.append(f'dc rrpfromxml {fin.pars_xml_path}')
        prev_folder = folder
    mac_fn = cluster_name + '_finalize.mac'

    if not _skip_mac_write:
        with open(mac_fn, 'w') as fh:
            fh.write('\n'.join(cmds))    

    print('Please run the following command in CAP:\n-----')
    print(f'script {mac_fn}')
    print('-----')
    print('...and set finalization parameters (Laue group, most importantly) for each cluster')

    if _write_mac_only:
        return mac_fn

    # %%
    # wait for template XML files and broadcast to XML for each node

    from time import sleep
    print('Waiting for template XML files...')

    # template_files = [(k, v) for k, v in template_files.items()]
    _templates = list(template_files.items())

    while _templates:
        for cluster, template_fn in _templates:
            if os.path.exists(template_fn):
                print(f'Finalization template for cluster {cluster} found:', template_fn)
                for _, fin in fc.items():
                    if fin.meta['Cluster'] == cluster:
                        fin.pars_xml.set_parameters(template=template_fn, 
                        autochem=False, gral=False, res_limit=res_limit)
                
                _templates.remove((cluster, template_fn))
        sleep(0.05)
            
    print('All finalization parameter files have been created. CAP should start with finalizations now...')

    # %% Parse finalizations
    from time import sleep
    fins_todo = list(fc.sort_by_meta(by=['Cluster', 'File path']).keys())

    while fins_todo:
        for fin_name in fins_todo:
            try:
                fc[fin_name].parse_finalization_results(check_current=True, timeout=finalization_timeout)
                print(f'Finalization for {fin_name} completed, results found')
                fins_todo.remove(fin_name)            
            except FileNotFoundError:
                pass
                
        sleep(0.5)

    print(f'Finalizations completed for {cluster_name}')

    return fc

if __name__ == '__main__':
    fc = cluster_finalize(
        cluster_name = 'C:\\XcaliburData\\Trehalose-Symmetry\\all-results'
    )
    
    print('OVERALL RESULTS TABLE')
    print('---------------------')
    print(fc.overall_highest)