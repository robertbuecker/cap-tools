"""
Minimal Python module for reading Rigaku Oxford Diffraction image files.

This module extracts the core functionality from dxtbx's FormatROD class which can be
found at:
https://github.com/cctbx/dxtbx

Original copyright and license information from dxtbx is retained below.
Authors: David Waterman, Takanori Nakane
Copyright: 2018-2023 United Kingdom Research and Innovation & 2022-2023 Takanori Nakane
License: BSD 3-clause
"""

import os
import re
import struct
import numpy as np
from typing import Dict, Tuple, Union, Optional

# Try to import the C++ decompression function from dxtbx if available
try:
    from dxtbx.ext import uncompress_rod_TY6 # pyright: ignore[reportMissingImports]
    HAS_CPP_DECOMPRESSION = True
except ImportError:
    HAS_CPP_DECOMPRESSION = False

# Try to import Numba for JIT acceleration
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# Numba-accelerated TY6 decompression functions
if HAS_NUMBA:
    @jit(nopython=True, cache=True) # pyright: ignore[reportPossiblyUnboundVariable]
    def _decode_ty6_oneline_numba(linedata: np.ndarray, w: int) -> np.ndarray:
        """
        Numba JIT-compiled version of TY6 line decompression.
        
        Args:
            linedata: Raw line data as uint8 array
            w: Number of pixels in the fast axis
            
        Returns:
            Decompressed pixel values for the line
        """
        BLOCKSIZE = 8
        SHORT_OVERFLOW = 254
        LONG_OVERFLOW = 255
        SHORT_OVERFLOW_SIGNED = SHORT_OVERFLOW - 127
        LONG_OVERFLOW_SIGNED = LONG_OVERFLOW - 127

        ipos = 0
        opos = 0
        ret = np.zeros(w, dtype=np.int32)

        nblock = (w - 1) // (BLOCKSIZE * 2)
        nrest = (w - 1) % (BLOCKSIZE * 2)

        # Decode first pixel
        firstpx = int(linedata[ipos])
        ipos += 1
        if firstpx < SHORT_OVERFLOW:
            ret[opos] = firstpx - 127
        elif firstpx == LONG_OVERFLOW:
            # Manually reconstruct int32 from bytes
            ret[opos] = (linedata[ipos] | 
                        (linedata[ipos + 1] << 8) | 
                        (linedata[ipos + 2] << 16) | 
                        (linedata[ipos + 3] << 24))
            # Handle signed integer overflow
            if ret[opos] >= 2147483648:
                ret[opos] -= 4294967296
            ipos += 4
        else:
            # Manually reconstruct int16 from bytes
            val = linedata[ipos] | (linedata[ipos + 1] << 8)
            # Handle signed integer overflow for int16
            if val >= 32768:
                val -= 65536
            ret[opos] = val
            ipos += 2
        opos += 1

        # Decode blocks
        for k in range(nblock):
            bittype = int(linedata[ipos])
            nbit1 = bittype & 15
            nbit2 = (bittype >> 4) & 15
            ipos += 1

            # Process first sub-block
            zero_at1 = 0
            if nbit1 > 1:
                zero_at1 = (1 << (nbit1 - 1)) - 1

            v1 = 0
            for j in range(nbit1):
                v1 |= int(linedata[ipos]) << (8 * j)
                ipos += 1

            mask1 = (1 << nbit1) - 1
            for j in range(BLOCKSIZE):
                val = ((v1 >> (nbit1 * j)) & mask1) - zero_at1
                ret[opos] = np.int32(val)
                opos += 1

            # Process second sub-block
            zero_at2 = 0
            if nbit2 > 1:
                zero_at2 = (1 << (nbit2 - 1)) - 1

            v2 = 0
            for j in range(nbit2):
                v2 |= int(linedata[ipos]) << (8 * j)
                ipos += 1

            mask2 = (1 << nbit2) - 1
            for j in range(BLOCKSIZE):
                val = ((v2 >> (nbit2 * j)) & mask2) - zero_at2
                ret[opos] = np.int32(val)
                opos += 1

            # Apply delta encoding to the entire block
            block_start = opos - BLOCKSIZE * 2
            for i in range(block_start, opos):
                offset = ret[i]

                if offset >= SHORT_OVERFLOW_SIGNED:
                    if offset >= LONG_OVERFLOW_SIGNED:
                        # Manually reconstruct int32 from bytes
                        offset_val = (linedata[ipos] | 
                                    (linedata[ipos + 1] << 8) | 
                                    (linedata[ipos + 2] << 16) | 
                                    (linedata[ipos + 3] << 24))
                        # Handle signed integer overflow
                        if offset_val >= 2147483648:
                            offset_val -= 4294967296
                        offset = offset_val
                        ipos += 4
                    else:
                        # Manually reconstruct int16 from bytes
                        val = linedata[ipos] | (linedata[ipos + 1] << 8)
                        # Handle signed integer overflow for int16
                        if val >= 32768:
                            val -= 65536
                        offset = val
                        ipos += 2

                ret[i] = ret[i - 1] + offset

        # Decode remaining pixels
        for i in range(nrest):
            px = int(linedata[ipos])
            ipos += 1
            if px < SHORT_OVERFLOW:
                ret[opos] = ret[opos - 1] + px - 127
            elif px == LONG_OVERFLOW:
                # Manually reconstruct int32 from bytes
                val = (linedata[ipos] | 
                      (linedata[ipos + 1] << 8) | 
                      (linedata[ipos + 2] << 16) | 
                      (linedata[ipos + 3] << 24))
                # Handle signed integer overflow
                if val >= 2147483648:
                    val -= 4294967296
                ret[opos] = ret[opos - 1] + val
                ipos += 4
            else:
                # Manually reconstruct int16 from bytes
                val = linedata[ipos] | (linedata[ipos + 1] << 8)
                # Handle signed integer overflow for int16
                if val >= 32768:
                    val -= 65536
                ret[opos] = ret[opos - 1] + val
                ipos += 2
            opos += 1

        return ret

    @jit(nopython=True, cache=True) # pyright: ignore[reportPossiblyUnboundVariable]
    def _decode_ty6_image_numba(linedata: np.ndarray, offsets: np.ndarray, 
                               ny: int, nx: int) -> np.ndarray:
        """
        Numba JIT-compiled version of full TY6 image decompression.
        
        Args:
            linedata: Raw compressed data as uint8 array
            offsets: Line offset positions as uint32 array
            ny: Number of lines (height)
            nx: Number of pixels per line (width)
            
        Returns:
            Decompressed image as 2D int32 array
        """
        image = np.zeros((ny, nx), dtype=np.int32)
        for iy in range(ny):
            line_start = offsets[iy]
            if iy < ny - 1:
                line_end = offsets[iy + 1]
                line_slice = linedata[line_start:line_end]
            else:
                line_slice = linedata[line_start:]
            image[iy, :] = _decode_ty6_oneline_numba(line_slice, nx)
        return image


class RODImageReader:
    """
    Minimal reader for Rigaku Oxford Diffraction image files.
    
    This class can read .rodhypix files and return the image data as NumPy arrays.
    It supports both C++ accelerated decompression (if dxtbx is available) and
    pure Python decompression.
    """
    
    def __init__(self, image_file: Union[str, os.PathLike], use_cpp: bool = True, use_numba: bool = True):
        """
        Initialize the reader with an image file.
        
        Args:
            image_file: Path to the .rodhypix file
            use_cpp: Use C++ decompression if available (default True)
            use_numba: Use Numba JIT decompression if available (default True, fallback from C++)
        """
        self.image_file = os.fspath(image_file)
        self.use_cpp = use_cpp and HAS_CPP_DECOMPRESSION
        self.use_numba = use_numba and HAS_NUMBA
        
        if not self.understand(self.image_file):
            raise ValueError(f"File {self.image_file} is not a valid ROD format")
            
        self._txt_header: Optional[Dict] = None
        self._bin_header: Optional[Dict] = None
        self._read_headers()
    
    @staticmethod
    def understand(image_file: Union[str, os.PathLike]) -> bool:
        """
        Check if the file is a valid Rigaku Oxford Diffraction format.
        
        Args:
            image_file: Path to the image file
            
        Returns:
            True if the file is a valid ROD format
        """
        try:
            with open(image_file, "rb") as f:
                hdr = f.read(256).decode("ascii")
        except (OSError, UnicodeDecodeError):
            return False

        lines = hdr.splitlines()
        if len(lines) < 2:
            return False

        vers = lines[0].split()
        if len(vers) < 2 or vers[0] != "OD" or vers[1] != "SAPPHIRE":
            return False

        compression = lines[1].split("=")
        if compression[0] != "COMPRESSION":
            return False

        return True
    
    def _read_headers(self) -> None:
        """Read both ASCII and binary headers."""
        self._txt_header = self._read_ascii_header()
        self._bin_header = self._read_binary_header()
    
    def _read_ascii_header(self) -> Dict:
        """Read the ASCII header comprising the first 256 bytes of the file."""
        hd = {}
        with open(self.image_file, "rb") as f:
            hdr = f.read(256).decode("ascii")
        lines = hdr.splitlines()

        vers = lines[0].split()
        if len(vers) < 2 or vers[0] != "OD" or vers[1] != "SAPPHIRE":
            raise ValueError("Wrong header format")
        hd["version"] = float(vers[-1])

        compression = lines[1].split("=")
        if compression[0] != "COMPRESSION":
            raise ValueError("Wrong header format")
        hd["compression"] = compression[1]

        # Extract definitions from the 3rd - 5th line
        defn = re.compile(r"([A-Z]+=[ 0-9]+)")
        for line in lines[2:5]:
            sizes = defn.findall(line)
            for s in sizes:
                n, v = s.split("=")
                hd[n] = int(v)

        hd["time"] = lines[5].split("TIME=")[-1].strip("\x1a").rstrip()

        return hd
    
    def _read_binary_header(self) -> Dict:
        """Read the most relevant parameters from the binary header."""
        offset = 256
        general_nbytes = 512
        special_nbytes = 768
        km4gonio_nbytes = 1024
        
        with open(self.image_file, "rb") as f:
            # General section
            f.seek(offset)
            bin_x, bin_y = struct.unpack("<hh", f.read(4))
            f.seek(offset + 22)
            chip_npx_x, chip_npx_y, im_npx_x, im_npx_y = struct.unpack(
                "<hhhh", f.read(8)
            )
            f.seek(offset + 36)
            num_points = struct.unpack("<I", f.read(4))[0]
            if num_points != im_npx_x * im_npx_y:
                raise ValueError("Cannot interpret binary header")

            # Special section
            f.seek(offset + general_nbytes + 56)
            gain = struct.unpack("<d", f.read(8))[0]
            f.seek(offset + general_nbytes + 464)
            overflow_flag, overflow_after_remeasure_flag = struct.unpack(
                "<hh", f.read(4)
            )
            f.seek(offset + general_nbytes + 472)
            overflow_threshold = struct.unpack("<l", f.read(4))[0]
            f.seek(offset + general_nbytes + 480)
            exposure_time_sec, overflow_time_sec = struct.unpack("<dd", f.read(16))
            f.seek(offset + general_nbytes + 548)
            detector_type = struct.unpack("<l", f.read(4))[0]
            f.seek(offset + general_nbytes + 568)
            real_px_size_x, real_px_size_y = struct.unpack("<dd", f.read(16))

            # Goniometer section
            f.seek(offset + general_nbytes + special_nbytes + 284)
            # angles for OMEGA, THETA, CHI(=KAPPA), PHI,
            # OMEGA_PRIME (also called DETECTOR_AXIS; what's this?), THETA_PRIME
            start_angles_steps = struct.unpack("<llllllllll", f.read(40))
            end_angles_steps = struct.unpack("<llllllllll", f.read(40))
            f.seek(offset + general_nbytes + special_nbytes + 368)
            step_to_rad = struct.unpack("<dddddddddd", f.read(80))
            f.seek(offset + general_nbytes + special_nbytes + 552)
            # FIXME: I don't know what these are. Isn't the beam along e1 by definition??
            beam_rotn_around_e2, beam_rotn_around_e3 = struct.unpack("<dd", f.read(16))
            alpha1_wavelength = struct.unpack("<d", f.read(8))[0]
            alpha2_wavelength = struct.unpack("<d", f.read(8))[0]
            alpha12_wavelength = struct.unpack("<d", f.read(8))[0]
            f.seek(offset + general_nbytes + special_nbytes + 640)
            # detector rotation in degrees along e1, e2, e3
            detector_rotns = struct.unpack("<ddd", f.read(24))
            # direct beam position when all angles are zero (FIXME: not completely sure)
            origin_px_x, origin_px_y = struct.unpack("<dd", f.read(16))
            # alpha and beta are angles between KAPPA(=CHI) and THETA, and e3.
            angles_in_deg = struct.unpack(
                "<dddd", f.read(32)
            )  # alpha, beta, gamma, delta
            f.seek(offset + general_nbytes + special_nbytes + 712)
            distance_mm = struct.unpack("<d", f.read(8))[0]

        return {
            "bin_x": bin_x,
            "bin_y": bin_y,
            "chip_npx_x": chip_npx_x,
            "chip_npx_y": im_npx_y,
            "im_npx_x": im_npx_x,
            "im_npx_y": im_npx_y,
            "gain": gain,
            "overflow_flag": overflow_flag,
            "overflow_after_remeasure_flag": overflow_after_remeasure_flag,
            "overflow_threshold": overflow_threshold,
            "exposure_time_sec": exposure_time_sec,
            "overflow_time_sec": overflow_time_sec,
            "detector_type": detector_type,
            "real_px_size_x": real_px_size_x,
            "real_px_size_y": real_px_size_y,
            "start_angles_steps": start_angles_steps,
            "end_angles_steps": end_angles_steps,
            "step_to_rad": step_to_rad,
            "beam_rotn_around_e2": beam_rotn_around_e2,
            "beam_rotn_around_e3": beam_rotn_around_e3,
            "alpha1_wavelength": alpha1_wavelength,
            "alpha2_wavelength": alpha2_wavelength,
            "alpha12_wavelength": alpha12_wavelength,
            "detector_rotns": detector_rotns,
            "origin_px_x": origin_px_x,
            "origin_px_y": origin_px_y,
            "angles_in_deg": angles_in_deg,
            "distance_mm": distance_mm,
        }
    
    def get_image_shape(self) -> Tuple[int, int]:
        """
        Get the shape of the image (height, width).
        
        Returns:
            Tuple of (ny, nx) where ny is height and nx is width
        """
        assert self._txt_header is not None
        return (self._txt_header["NY"], self._txt_header["NX"])
    
    def get_pixel_size(self) -> Tuple[float, float]:
        """
        Get the pixel size in mm.
        
        Returns:
            Tuple of (pixel_size_x, pixel_size_y) in mm
        """
        assert self._bin_header is not None
        return (self._bin_header["real_px_size_x"], self._bin_header["real_px_size_y"])
    
    def get_exposure_time(self) -> float:
        """
        Get the exposure time in seconds.
        
        Returns:
            Exposure time in seconds
        """
        assert self._bin_header is not None
        return self._bin_header["exposure_time_sec"]
    
    def get_raw_data(self) -> np.ndarray:
        """
        Read the image data and return as a NumPy array.
        
        Returns:
            2D NumPy array with the image data
        """
        assert self._txt_header is not None
        comp = self._txt_header["compression"].strip()
        if comp.startswith("TY6"):
            if self.use_cpp:
                return self._get_raw_data_ty6_cpp()
            elif self.use_numba:
                return self._get_raw_data_ty6_numba()
            else:
                return self._get_raw_data_ty6_python()
        else:
            raise NotImplementedError(f"Can't handle compression: {comp}")
    
    def _get_raw_data_ty6_cpp(self) -> np.ndarray:
        """Read TY6 compressed data using C++ decompression."""
        if not HAS_CPP_DECOMPRESSION:
            raise RuntimeError("C++ decompression not available")
            
        assert self._txt_header is not None
        offset = self._txt_header["NHEADER"]
        nx = self._txt_header["NX"]
        ny = self._txt_header["NY"]
        
        with open(self.image_file, "rb") as f:
            f.seek(offset)
            lbytesincompressedfield = struct.unpack("<l", f.read(4))[0]
            linedata = f.read(lbytesincompressedfield)
            offsets = f.read(4 * ny)

            # Import flex here to avoid issues if not available
            from scitbx.array_family import flex # pyright: ignore[reportMissingImports]
            flex_result = uncompress_rod_TY6(linedata, offsets, ny, nx)  # type: ignore
            return flex_result.as_numpy_array()
    
    def _get_raw_data_ty6_python(self) -> np.ndarray:
        """Read TY6 compressed data using pure Python decompression."""
        assert self._txt_header is not None
        offset = self._txt_header["NHEADER"]
        nx = self._txt_header["NX"]
        ny = self._txt_header["NY"]
        
        with open(self.image_file, "rb") as f:
            f.seek(offset)
            lbytesincompressedfield = struct.unpack("<l", f.read(4))[0]
            linedata = np.fromfile(f, dtype=np.uint8, count=lbytesincompressedfield)
            offsets = struct.unpack("<%dI" % ny, f.read(4 * ny))

            image = np.zeros((ny, nx), dtype=np.int32)
            for iy in range(ny):
                image[iy, :] = self._decode_ty6_oneline(linedata[offsets[iy]:], nx)
            
            return image
    
    def _decode_ty6_oneline(self, linedata: np.ndarray, w: int) -> np.ndarray:
        """
        Decompress TY6 encoded pixels for a single line.
        
        Args:
            linedata: Raw line data as uint8 array
            w: Number of pixels in the fast axis
            
        Returns:
            Decompressed pixel values for the line
        """
        BLOCKSIZE = 8
        SHORT_OVERFLOW = 254
        LONG_OVERFLOW = 255
        SHORT_OVERFLOW_SIGNED = SHORT_OVERFLOW - 127
        LONG_OVERFLOW_SIGNED = LONG_OVERFLOW - 127

        ipos = 0
        opos = 0
        ret = np.zeros(w, dtype=np.int32)

        nblock = (w - 1) // (BLOCKSIZE * 2)
        nrest = (w - 1) % (BLOCKSIZE * 2)

        # Decode first pixel
        firstpx = int(linedata[ipos])
        ipos += 1
        if firstpx < SHORT_OVERFLOW:
            ret[opos] = firstpx - 127
        elif firstpx == LONG_OVERFLOW:
            ret[opos] = linedata[ipos:(ipos + 4)].view(np.int32)[0]
            ipos += 4
        else:
            ret[opos] = linedata[ipos:(ipos + 2)].view(np.int16)[0]
            ipos += 2
        opos += 1

        # Decode blocks
        for k in range(nblock):
            bittype = int(linedata[ipos])
            nbits = (bittype & 15, (bittype >> 4) & 15)
            ipos += 1

            for i in range(2):
                nbit = nbits[i]
                zero_at = 0
                if nbit > 1:
                    zero_at = (1 << (nbit - 1)) - 1

                v = 0
                for j in range(nbit):
                    v |= int(linedata[ipos]) << (8 * j)
                    ipos += 1

                mask = (1 << nbit) - 1
                for j in range(BLOCKSIZE):
                    val = ((v >> (nbit * j)) & mask) - zero_at
                    ret[opos] = np.int32(val)
                    opos += 1

            # Apply delta encoding
            for i in range(opos - BLOCKSIZE * 2, opos):
                offset = ret[i]

                if offset >= SHORT_OVERFLOW_SIGNED:
                    if offset >= LONG_OVERFLOW_SIGNED:
                        offset = linedata[ipos:(ipos + 4)].view(np.int32)[0]
                        ipos += 4
                    else:
                        offset = linedata[ipos:(ipos + 2)].view(np.int16)[0]
                        ipos += 2

                ret[i] = ret[i - 1] + offset

        # Decode remaining pixels
        for i in range(nrest):
            px = int(linedata[ipos])
            ipos += 1
            if px < SHORT_OVERFLOW:
                ret[opos] = ret[opos - 1] + px - 127
            elif px == LONG_OVERFLOW:
                ret[opos] = (
                    ret[opos - 1] + linedata[ipos:(ipos + 4)].view(np.int32)[0]
                )
                ipos += 4
            else:
                ret[opos] = (
                    ret[opos - 1] + linedata[ipos:(ipos + 2)].view(np.int16)[0]
                )
                ipos += 2
            opos += 1

        return ret
    
    def _get_raw_data_ty6_numba(self) -> np.ndarray:
        """Read TY6 compressed data using Numba-accelerated Python decompression."""
        if not HAS_NUMBA:
            raise RuntimeError("Numba not available")
            
        assert self._txt_header is not None
        offset = self._txt_header["NHEADER"]
        nx = self._txt_header["NX"]
        ny = self._txt_header["NY"]
        
        with open(self.image_file, "rb") as f:
            f.seek(offset)
            lbytesincompressedfield = struct.unpack("<l", f.read(4))[0]
            linedata = np.fromfile(f, dtype=np.uint8, count=lbytesincompressedfield)
            offsets_raw = f.read(4 * ny)
            offsets = np.frombuffer(offsets_raw, dtype=np.uint32)

            return _decode_ty6_image_numba(linedata, offsets, ny, nx)
    
    def get_decompression_method(self) -> str:
        """
        Get the decompression method that will be used.
        
        Returns:
            String indicating the decompression method: 'C++', 'Numba', or 'Python'
        """
        if self.use_cpp:
            return "C++"
        elif self.use_numba:
            return "Numba"
        else:
            return "Python"
    
    def get_header_info(self) -> Dict:
        """
        Get a summary of header information.
        
        Returns:
            Dictionary with key header information
        """
        assert self._txt_header is not None
        assert self._bin_header is not None
        # return {
        #     "version": self._txt_header["version"],
        #     "compression": self._txt_header["compression"],
        #     "image_shape": self.get_image_shape(),
        #     "pixel_size_mm": self.get_pixel_size(),
        #     "exposure_time_sec": self.get_exposure_time(),
        #     "gain": self._bin_header["gain"],
        #     "overflow_threshold": self._bin_header["overflow_threshold"],
        #     "timestamp": self._txt_header["time"],
        # }
        all_meta = {**self._txt_header, **self._bin_header}

        return all_meta


# Convenience functions
def read_rod_image(filename: Union[str, os.PathLike], 
                   use_cpp: bool = True, use_numba: bool = True) -> np.ndarray:
    """
    Read a ROD image file and return the data as a NumPy array.
    
    Args:
        filename: Path to the .rodhypix file
        use_cpp: Use C++ decompression if available (default True)
        use_numba: Use Numba JIT decompression if available (default True, fallback from C++)
        
    Returns:
        2D NumPy array with the image data
    """
    reader = RODImageReader(filename, use_cpp=use_cpp, use_numba=use_numba)
    return reader.get_raw_data()


def get_rod_info(filename: Union[str, os.PathLike]) -> Dict:
    """
    Get header information from a ROD image file.
    
    Args:
        filename: Path to the .rodhypix file
        
    Returns:
        Dictionary with header information
    """
    reader = RODImageReader(filename)
    return reader.get_header_info()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python minimal_rod_reader.py <image_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        # Try with C++ decompression first
        print(f"Reading {filename}...")
        reader = RODImageReader(filename)
        
        print("Header info:")
        info = reader.get_header_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print(f"\nDecompression method: {'C++' if reader.use_cpp else 'Python'}")
        
        data = reader.get_raw_data()
        print(f"Image shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean():.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)