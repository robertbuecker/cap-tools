# Computation of absolute structure determination Z-scores
# Following Klar et al., doi:10.1038/s41557-023-01186-1
# Written 2022 by Sho Ito
# Extended 2023 by Robert Bücker

import math
from collections import namedtuple
from typing import Dict

Reflection = namedtuple('Reflection', ['h', 'k', 'l', 'Icalc', 'Iobs', 'sigma', 'group', 'status'])
RefCompare = namedtuple('ReflectionCompare',['R_assign', 'p_error', 'Z_single'])

def parse_reflection_file(fn: str) -> Dict[tuple, Reflection]:
    """ _refln_index_h
 _refln_index_k
 _refln_index_l
 _refln_intensity_calc
 _refln_intensity_meas
 _refln_intensity_sigma
 _refln_scale_group_code
 _refln_observed_status
    """    
    parsing = False
    ref_list = {}
    with open(fn) as fh:
        for ii, ln in enumerate(fh):
            if parsing and (ln == '\n'):
                parsing = False
                print(f'to line {ii}')
                break
            elif parsing:
                f = ln.split()
                ref_list[(int(f[0]), int(f[1]), int(f[2]))] = \
                    Reflection(h=int(f[0]), k=int(f[1]), l=int(f[2]),
                               Icalc=float(f[3]), Iobs=float(f[4]),
                               sigma=float(f[5]), group=int(f[6]), status=f[7])
                
            if ln.strip() == '_refln_observed_status':
                print(f'Parsing reflection list in {fn} from line {ii+1} ', end='')
                parsing = True
    return ref_list

def compare_reflections(R: Dict[tuple, Reflection], S: Dict[tuple, Reflection]) -> Dict[tuple, RefCompare]:
    all_hkl = set((*R.keys(), *S.keys()))
    comparison_data = {}
    warnings = []
    
    for hkl in all_hkl:
        # run sanity checks
        try:
            r_ref = R[hkl]
        except KeyError as err:
            warnings.append(f'Missing reflection in first file (R): {hkl}')
            continue
        try:
            s_ref = S[hkl]
        except KeyError as err:
            warnings.append(f'Missing reflection in second file (S): {hkl}')
            continue
        if (r_ref.Iobs != s_ref.Iobs) or (r_ref.sigma != s_ref.sigma):
            warnings.append(f'Inconsistent observed intensity or sigma in {hkl}')
            continue
        if r_ref.group != s_ref.group:
            warnings.append(f"Inconsistent scaling group for {hkl}: R={r_ref.group} vs S={s_ref.group}")
        if r_ref.status != s_ref.status:
            warnings.append(f"Inconsistent status for {hkl}: R={r_ref.status} vs S={s_ref.status}")
             
        if (r_ref.status == 'o') and (s_ref.status == 'o'):
            comparison_data[hkl] = RefCompare(
                R_assign=abs(r_ref.Iobs-r_ref.Icalc) < abs(s_ref.Iobs-s_ref.Icalc),
                p_error=1 - math.erf(abs(r_ref.Icalc-s_ref.Icalc)/(2*r_ref.sigma)),
                Z_single=(r_ref.Icalc-s_ref.Icalc)/r_ref.sigma
            )

    if warnings:
        print('----- WARNINGS DURING FILE COMPARISON -----')
        print('\n'.join(warnings)) 
        print(f'----- {len(warnings)} WARNINGS TOTAL -----')
        
        
    return comparison_data

def calc_assignment(cif_correct_hand: str, cif_wrong_hand: str) -> tuple:

    correct = parse_reflection_file(cif_correct_hand)
    wrong = parse_reflection_file(cif_wrong_hand)
    
    comp = compare_reflections(R=correct, S=wrong)
    
    return(len(comp),
           sum([c.R_assign for c in comp.values()]), 
           sum([c.p_error for c in comp.values()]))
        
def calc_z(N, k, w=None): 
    
    """Be N the
number of reflections in the data set and be k the number of reflections for which |Icalc,1竏棚obs| <
|Icalc,2竏棚obs|, i.e., for which the enantiomorph labelled 1 gives a better match to the observed data.
1 means correct hand.
If w (number of estimated mis-assignments) is supplied, the adjusted z-score is used, according to 
Klar et al., 10.1038/s41557-023-01186-1
        """
    if w is not None:
        return (2*k-N)/math.sqrt(N-w)
    else:
        return (k-(N/2))/(math.sqrt(N)/2)
    
def main():
    C = sys.argv[1]
    W = sys.argv[2]
    
    print('-----')
    print('Absolute refinement Z-score computation')
    print('Please cite: Klar et al., Nature Chemistry 15 (2023), 848. doi:10.1038/s41557-023-01186-1')
    
    print(f'C: {C}\nW: {W} ')

    N, k, w = calc_assignment(C, W)
    z_score = calc_z(N, k, w)
    z_score_raw = calc_z(N, k, None)
    print(f'{N} observations, {k} ({k/N*100:.1f}%) have bias for C, {w:.0f} expected mis-assignments' 
          f'\n-> noise-adjusted Z-score is {z_score:.2f} (non-adjusted {z_score_raw:.2f})')
    
def gui():
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter.filedialog import askopenfilename
    
    root = tk.Tk()
    root.title('Absolute refinement Z-score computation')
    
    fns = {'R': tk.StringVar(), 'S': tk.StringVar()}
    
    result_text = tk.StringVar(value='Please enter filenames and press "Compute"\n')
    
    def set_fn(which: str):
        fn = askopenfilename(title=f'Open file for {which} config', filetypes=[('JANA dynamical refinement result', '*.cif')])
        fns[which].set(fn)
        
    def compute():
        N, k, w = calc_assignment(fns['R'].get(), fns['S'].get())
        z_score = calc_z(N, k, w)
        z_score_raw = calc_z(N, k, None)
        result_text.set(f'{N} observations, {k} ({k/N*100:.1f}%) have bias for R, {w:.0f} expected mis-assignments due to noise' 
            f'\n-> noise-adjusted Z-score (positive for R) is {z_score:.2f} (non-adjusted {z_score_raw:.2f})')        
    
    ttk.Label(root, text='Please cite: Klar et al., Nature Chemistry 15 (2023), 848. doi:10.1038/s41557-023-01186-1').grid(row=0, column=0, columnspan=2)
    ttk.Button(root, text='Open R', command=lambda which='R': set_fn(which)).grid(row=5, column=0, sticky=tk.W)
    ttk.Button(root, text='Open S', command=lambda which='S': set_fn(which)).grid(row=10, column=0, sticky=tk.W)
    ttk.Label(root, textvariable=fns['R']).grid(row=5, column=1, sticky=tk.W)
    ttk.Label(root, textvariable=fns['S']).grid(row=10, column=1, sticky=tk.W)
    ttk.Button(root, text='Compute', command=compute).grid(row=15, column=0, columnspan=2)
    ttk.Separator(root, orient='horizontal').grid(row=19, columnspan=2, sticky=tk.EW)
    ttk.Label(root, textvariable=result_text).grid(row=20, column=0, columnspan=2)
    
    tk.mainloop()
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        gui()
    