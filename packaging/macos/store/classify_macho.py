#!/usr/bin/env python3
"""Lista los Mach-O de un bundle: 'exec<TAB>ruta' (ejecutables) o 'lib<TAB>ruta' (dylib, .so, bundles).
Uso: classify_macho.py <dir>   Sale ordenado de lo más profundo a lo menos (para firmar de adentro hacia afuera)."""
import os, struct, sys

MH_MAGIC_64 = 0xFEEDFACF
FAT = (0xCAFEBABE, 0xBEBAFECA)

def kind(path):
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return None
    if len(head) < 16:
        return None
    magic = struct.unpack("<I", head[:4])[0]
    if magic == MH_MAGIC_64:
        ftype = struct.unpack("<I", head[12:16])[0]
        return "exec" if ftype == 2 else "lib"
    if struct.unpack(">I", head[:4])[0] in FAT:
        return "lib"  # binarios universales: no los hay en el bundle de Apple Silicon; se firman como librería
    return None

out = []
for root, dirs, files in os.walk(sys.argv[1]):
    for n in files:
        p = os.path.join(root, n)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        k = kind(p)
        if k:
            out.append((k, p))
out.sort(key=lambda t: (-t[1].count("/"), t[1]))
for k, p in out:
    print(f"{k}\t{p}")
