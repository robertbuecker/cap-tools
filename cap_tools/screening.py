"""Backend helpers for screening workflows.

This module is the stable boundary for the staged screening-viewer refactor.
The current viewer can keep its Tk layout while analysis code moves here in
small, testable steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cap_auto.cap_control import CAPInstance
from cap_auto.cap_files import parse_cap_meta
from peakfinder8 import peakfinder8

from cap_tools.process import get_diff_info


@dataclass(frozen=True)
class Peakfinder8Options:
    max_peaks: int = 2000
    min_snr: float = 4.0
    threshold: float = 40.0
    min_pix_count: int = 1
    max_pix_count: int = 40
    local_bg_radius: int = 3
    min_res: int = 0
    max_res: int = 4096


def load_experiment_metadata(
    experiments: List[str],
    *,
    pre_only: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    include = ['pre_'] if pre_only else None
    rows = parse_cap_meta(experiments, include=include, log_fun=log)
    frame = pd.DataFrame(rows)
    if not frame.empty and 'name' in frame.columns:
        frame.set_index('name', inplace=True)
    return frame


def analyze_experiment_with_cap(
    exp_path: str,
    cap: CAPInstance,
    *,
    keep_existing: bool = True,
    redo_peak_hunt: bool = True,
    recenter_pattern: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    return get_diff_info(
        exp_path,
        cap=cap,
        keep_peak_file=keep_existing,
        keep_powder_file=keep_existing,
        redo_peak_hunt=redo_peak_hunt,
        recenter_pattern=recenter_pattern,
        log=log,
    )


def find_peaks_raw_frame(
    image: np.ndarray,
    *,
    options: Peakfinder8Options = Peakfinder8Options(),
    center: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Run peakfinder8 on one raw frame and return a normalized peak table."""
    data = image.astype(np.float32, copy=False)
    y0, x0 = center if center is not None else (data.shape[0] / 2.0, data.shape[1] / 2.0)
    yy, xx = np.indices(data.shape, dtype=np.float32)
    radius = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2).astype(np.float32)
    mask = np.ones_like(data, dtype=np.int8)
    mask[radius > options.max_res] = 0
    mask[radius < options.min_res] = 0

    peaks = peakfinder8(
        options.max_peaks,
        data,
        mask,
        radius,
        data.shape[1],
        data.shape[0],
        1,
        1,
        options.threshold,
        options.min_snr,
        options.min_pix_count,
        options.max_pix_count,
        options.local_bg_radius,
    )

    columns = [
        'x',
        'y',
        'I',
        'peak_index',
        'npix',
        'max_intensity',
        'sigma',
        'snr',
    ]
    return pd.DataFrame({column: values for column, values in zip(columns, peaks)})
