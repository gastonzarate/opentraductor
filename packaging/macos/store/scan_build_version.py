#!/usr/bin/env python3
"""Revisa LC_BUILD_VERSION de cada Mach-O: Apple valida plataforma, minos y SDK al subir a la Mac App Store.
Uso: scan_build_version.py <dir>   -> resumen y lista de binarios sin LC_BUILD_VERSION o con SDK viejo (< 10.9)."""
import collections, os, struct, sys

def info(path):
    with open(path, "rb") as f:
        d = f.read(1 << 16)
    if len(d) < 32 or struct.unpack("<I", d[:4])[0] != 0xFEEDFACF:
        return None
    ncmds, = struct.unpack("<I", d[16:20]); off = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack("<II", d[off:off + 8])
        if cmd == 0x32:  # LC_BUILD_VERSION
            plat, minos, sdk = struct.unpack("<III", d[off + 8:off + 20])
            v = lambda x: f"{x >> 16}.{(x >> 8) & 255}"
            return ("build", plat, v(minos), v(sdk))
        if cmd == 0x24:  # LC_VERSION_MIN_MACOSX (formato viejo)
            minos, sdk = struct.unpack("<II", d[off + 8:off + 16])
            return ("legacy", 1, f"{minos >> 16}.{(minos >> 8) & 255}", f"{sdk >> 16}.{(sdk >> 8) & 255}")
        off += size
    return ("none",)

c = collections.Counter(); bad = []
for r, _, fs in os.walk(sys.argv[1]):
    for n in fs:
        p = os.path.join(r, n)
        if os.path.islink(p): continue
        try: i = info(p)
        except OSError: continue
        if i is None: continue
        c[i[0] + (" minos " + i[2] + " sdk " + i[3] if len(i) > 2 else "")] += 1
        if i[0] != "build": bad.append((i[0], p))
for k, v in c.most_common(): print(f"{v:4d}  {k}")
print("sin LC_BUILD_VERSION:", len(bad))
for k, p in bad[:15]: print("  ", k, p)
