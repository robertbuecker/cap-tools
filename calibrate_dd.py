import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.registration import phase_cross_correlation
from skimage.filters import sato
from glob import glob
import os
from scipy.optimize import minimize, curve_fit
import sys
from matplotlib.patches import Ellipse

# Calibrant data Aluminum        
d_vec = np.array([2.338, 2.024, 1.431, 1.221, 1.169, 1.0124, 0.9289, 0.9055, 0.8266])
I_rel = np.array([100, 47, 22, 24, 7, 2, 8, 8, 8])
def process_pattern(pattern):
    # Pre-process pattern. Sato filtering works very well.
    return sato(pattern, black_ridges=False, sigmas=[1])


def find_center(pattern, show_fig = True, use_log = True, upsample = 100, suppress_radius = None, scale_radius = False):
    # Find pattern center by cross-correlating with its own flipped version
    # optionally, use a logarithm and/or suppress a central region around the direct beam
    
    if use_log:
        ptr = np.log10(pattern + 2)
    else:
        ptr = pattern
    if suppress_radius is not None:
        X, Y = np.meshgrid(np.arange(ptr.shape[1])-ptr.shape[1]/2,
                   np.arange(ptr.shape[0])-ptr.shape[0]/2)
        R = (X**2 + Y**2)**.5
        ptr = ptr*(1-np.exp(-(R/suppress_radius)**4))
    if scale_radius: ptr = ptr * R
    ctr2, *_ = phase_cross_correlation(ptr, np.rot90(ptr,2), upsample_factor=upsample)
    return ctr2/2

def radial_profile(pattern, ctr = (0,0), pxmap = None, stepsize = 0.5):
    # Compute the radial profile of the pattern with an optional dead pixel mask (set ignored pixels to 0, all others to 1)
    # Computation using weighted histogram of pixel radius values

    # 5, 4, 2, 1.5
    # (0, 1, 2, 3, 4) -> (-2, -1, 0, 1, 2); (0, 1, 2, 3) -> (-1.5, -0.5, +0.5, +1.5)
    Nx, Ny, x0, y0 = pattern.shape[1], pattern.shape[0], pattern.shape[1]/2-0.5 + ctr[1], pattern.shape[0]/2-0.5 + ctr[0]
    pxmap = np.ones_like(pattern) if pxmap is None else pxmap    
    X, Y = np.meshgrid(np.arange(Nx)-x0, np.arange(Ny)-y0)
    R, theta = (X**2 + Y**2)**.5, np.atan2(Y, X)*180/np.pi
    
    r_bin = np.arange(0, np.max(R.ravel()), stepsize) # set step size here
    num_pix, _ = np.histogram(R.ravel(), bins=r_bin, weights=pxmap.ravel())
    raw_sum, _ = np.histogram(R.ravel(), bins=r_bin, weights=(pxmap*pattern).ravel())
    r = (r_bin[:-1] + r_bin[1:])/2   # double-check this
    # r = r_bin[:-1]
    corr_sum = raw_sum/num_pix * np.pi*r
    
    return corr_sum, r
    
def spec(r, w_px=1., dd=1000., shift=0., lmb=0.0251, pxs=0.1):
    # Returns the expected radial spectrum in pixel units with peak width defined by w_px
    
    spec = np.zeros_like(r)
    r_rings = 2*np.sin(np.atan(lmb/d_vec)/2)*dd/pxs # doing the Bragg law properly - wow!
    for I, rr in zip(I_rel, r_rings):
        spec += I / w_px * np.exp(-((r-shift) - rr)**2/w_px**2)
    return spec

def overlap(spec_data, r, w_px=1., dd=1000., shift=0., lmb=0.0251, pxs=0.1, margin=20, normalized=True):
    # Returns overlap between a computed spectrum and the observed data
    rng = ((lmb/max(d_vec)*dd/pxs-margin) < r) & (r < (lmb/min(d_vec)*dd/0.1+margin))    
    if normalized:
        return np.nansum(spec(r, w_px, dd, shift, lmb, pxs)[rng]*spec_data[rng])/ \
            (np.sum(spec(r, w_px, dd, shift, lmb, pxs)[rng]**2)*np.nansum(spec_data[rng]**2))**.5
    else:
        return np.nansum(spec(r, w_px, dd, shift, lmb, pxs)[rng]*spec_data[rng])
    
def polar_pattern(pattern, ctr = (0,0), n_theta = 30, pxmap = None, stepsize = 0.5):
    # Computes the diffraction pattern in R-theta representation. Works similarly to radial_profile, but with 2D histograms

    Nx, Ny, x0, y0 = pattern.shape[1], pattern.shape[0], pattern.shape[1]/2-0.5 + ctr[1], pattern.shape[0]/2-0.5 + ctr[0]
    pxmap = np.ones_like(pattern) if pxmap is None else pxmap    
    X, Y = np.meshgrid(np.arange(Nx)-x0, np.arange(Ny)-y0)
    R, theta = (X**2 + Y**2)**.5, np.atan2(Y, X)*180/np.pi        
    r_bin = np.arange(0, np.max(R.ravel()), 1) # set step size here
    theta_bin = np.linspace(-180, 180, n_theta)
    
    num_pix_2d, _, _ = np.histogram2d(R.ravel(), theta.ravel(), bins=(r_bin, theta_bin), weights=pxmap.ravel())
    raw_sum_2d, _, _ = np.histogram2d(R.ravel(), theta.ravel(), bins=(r_bin, theta_bin), weights=(pxmap*pattern*R).ravel())
    corr_sum_2d = raw_sum_2d/num_pix_2d
    r = (r_bin[:-1] + r_bin[1:])/2   # double-check this
    theta = (theta_bin[1:] + theta_bin[:-1])/2
    
    return corr_sum_2d, r_bin, theta_bin, r, theta

def main(basedir: str):

    # step 1: make CAP script to create PETS files and TIF frames, if they are not there yet
    cmds = []
    cap_files = glob(os.path.join(basedir, '**\\*.par'), recursive=True)
    cap_files = [cf for cf in cap_files if not cf.endswith('_cracker.par')]
    pets_files = []
    for cf in cap_files:
        dir, lbl = os.path.dirname(cf), os.path.basename(cf).rsplit('.',1)[0]
        pets_file = glob(os.path.join(dir, '**\\*.pts2'), recursive=True)
        if not pets_file:
            cmds.append(f'xx selectexpnogui_ignoreerror "{cf}"')
            cmds.append(f'DC IMGTOPETS "{dir}\\frames\\PETS_{lbl}\\frames" 0 1 0 1 0')
        else:
            pets_files.extend(pets_file)

    if cmds:
        fn = os.path.join(basedir, 'export_tif.mac')
        with open(fn, 'w') as fh:
            fh.write('\n'.join(cmds))
        print('Not all calibration data found in PETS/TIF format. Please run:')
        print(f'SCRIPT {fn}')
        print(f'in CrysAlisPro and re-run this script. Press Enter quit.')
        input()
        exit()

    dd0 = {}
    imgs = {}

    # now iterate through pets files to get metadata
    for fn in pets_files:
        folder = os.path.dirname(fn)
        label =  os.path.basename(fn).rsplit('.',1)[0]                  
        with open(fn) as fh:
            for l in fh:
                if l.startswith('lambda'):
                    lmbd = float(l.strip().split()[-1])
                elif l.startswith('aperpixel'):
                    apix = float(l.strip().split()[-1])
                elif '.tif' in l:
                    imgs[label] = \
                        imread(os.path.join(folder, l.split()[0]))
            dd = 0.1/(lmbd*apix)
            dd0[label] = dd
            print(f'Loaded set {label} with current DD = {dd:.1f}')

    # and get to the real action
    for (k, img) in imgs.items():        
        pattern = img[15:-15, 15:-15] # offsetting a bit on purpose to stress out the correction algorithm a bit
        dd_init = dd0[k]
        pattern_proc = process_pattern(pattern)
        ctr = find_center(pattern_proc, use_log=False, suppress_radius=50, scale_radius=False) + [0, 0]
        # ctr = find_center(pattern_proc, use_log=True) #+ [0, -0.4]
        
        # overall detector distance
        pxmap = np.ones_like(pattern)
        prof, r = radial_profile(pattern_proc, ctr=ctr)
        min_res = minimize(lambda dd: 1-overlap(prof, r, w_px=1, dd=dd), [dd_init])
        dd_final = min_res.x[0]
        
        # per-slice distance and ellipticity
        r_d = 0.0251/d_vec*dd_final/0.1
        
        polar, r_bin2, r_theta2, r2, theta2 = polar_pattern(pattern_proc, ctr=ctr)
        dd2 = []
        for (prof2, th) in zip(polar.T, theta2):
            min_res = minimize(lambda dd: 1-overlap(prof2, r2, w_px=1, dd=dd), [dd_final])
            dd2.append(min_res.x[0])
        dd2 = np.array(dd2)
        dd2_span = max((dd_final-min(dd2), max(dd2)-dd_final))
        
        init_vals = [theta2[theta2>=0][np.argmax(dd2[theta2>=0])], dd2_span, dd2.mean()]
        ellipticity_fun = lambda x, ang, span, avg: avg + span/2*np.cos(2*np.pi/180*(x-ang))
        res, cov = curve_fit(ellipticity_fun, theta2, dd2, p0 = init_vals)
        
        print(f'RESULTS FOR {k} -------')
        print(f'DD from radial profile is {dd_final:.1f} mm. Center offset x={ctr[1]}, y={ctr[0]}')
        print(f'Average segment DD is {res[2]:.1f} mm with ellipticity {res[1]/res[2]*100:.2f}%, long axis at {res[0]:.1f} deg.')
        
        rng = (min(r_d-20) < r) & (r < (max(r_d)+20))
        rng2 = (min(r_d)-20 < r2) & (r2 < (max(r_d)+20))
        
        fh = plt.figure(figsize=(1.4*8.3, 1.4*11.7), dpi=300)
        
        axs = [fh.add_subplot(5, 1, 1), fh.add_subplot(5, 1, 2)]
        axs[0].imshow(pattern, vmax=np.percentile(pattern, 95), cmap='grey', 
                    extent=(-pattern.shape[1]/2, pattern.shape[1]/2, -pattern.shape[0]/2, pattern.shape[0]/2), origin='lower')
        axs[1].imshow(pattern_proc, vmax=np.percentile(pattern, 15), cmap='grey',
                    extent=(-pattern.shape[1]/2, pattern.shape[1]/2, -pattern.shape[0]/2, pattern.shape[0]/2), origin='lower')
        for rr in r_d:
            for ax in (axs[0], axs[1]):
                c = plt.Circle((ctr[1], ctr[0]), rr, ec='g', fill=False, alpha=0.8, lw=0.4)
                e = Ellipse((ctr[1], ctr[0]), 
                            width=rr*(res[2]+res[1]/2)/dd_final*2, 
                            height=rr*(res[2]-res[1]/2)/dd_final*2, 
                            angle=res[0], ec='r', fill=False, alpha=0.8, lw=0.4)
                ax.add_patch(c)
                ax.add_patch(e)
                
        axs[0].set_title(f'{k}: DD {dd_final:.1f}, offset ({ctr[1]:.1f}, {ctr[0]:.1f})')
        
        # fh, ax = plt.subplots(1,1, figsize=(15,3))
        ax = fh.add_subplot(5,1,3)
        ax.plot(r[rng], prof[rng])
        ax2 = ax.twinx()
        for rr in r_d:
            plt.axvline(rr, color='y')
        
        th_ax = np.linspace(-180, 180, 100)
        
        ax = fh.add_subplot(5, 1, 4) 
        ax.pcolormesh(r_bin2, r_theta2, polar.T, vmax=np.nanpercentile(polar, 90), cmap='gray')
        for rr in r_d:
            ax.axvline(rr, color='g', alpha=0.5)
            ax.plot(rr/dd_final*ellipticity_fun(th_ax, *res), th_ax, color='r', alpha=0.5)
        ax.set_ylabel('Azimuth [deg]')
        ax.set_xlabel('Radius [px]')
        
        ax = fh.add_subplot(5, 2, 9, projection='polar')
        ax.plot(theta2/180*np.pi, dd2, '--x')
        ax.plot(th_ax/180*np.pi, ellipticity_fun(th_ax, *res))
        ax.set_thetalim(-np.pi, np.pi)
        ax.set_rmax(dd_final + dd2_span)
        ax.set_rmin(dd_final - dd2_span)
        ax.set_rticks([min(dd2), dd_final, max(dd2)])
        ax = plt.subplot(5, 2, 10)
        ax.plot(theta2, dd2, '--x')
        ax.plot(th_ax, ellipticity_fun(th_ax, *res))
        # ax.plot(th_ax, ellipticity_fun(th_ax, *init_vals))
        ax.axhline(dd_final, c='y')
        ax.axhline(res[2], c='r')
        ax.set_title(f'Ellipticity {res[1]/res[2]*100:.1f}% at {res[0]:.1f} degrees. Avg DD = {res[2]:.1f}')
        
        plt.savefig(os.path.join(basedir, k + '.pdf'))

if __name__ == '__main__':
    main(sys.argv[1])