"""Lossless NumPy archives with an explicit, fast compression level."""
import re
import zipfile

import numpy as np


def write_npz(stream, arrays, *, compresslevel=1):
    if compresslevel != 1 or not arrays:
        raise ValueError("This registered writer requires nonempty arrays and DEFLATE level1")
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=compresslevel, allowZip64=True) as archive:
        for key, value in arrays.items():
            if not isinstance(key,str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",key):
                raise ValueError("Array names must be simple identifiers")
            array=np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError("Object arrays are forbidden")
            with archive.open(key+".npy",mode="w",force_zip64=True) as member:
                np.lib.format.write_array(member,array,allow_pickle=False)
