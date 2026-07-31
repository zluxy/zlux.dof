"""Stdlib-only replacement for the numpy/Pillow bits of prep.py.

The machine this repo is developed on no longer has numpy or Pillow, and the
harness only ever needs two things: write an ARGB8 .raw, and turn one back into
a viewable PNG. Both are a few lines of zlib + struct, so there is no reason to
put a dependency in the way of running a test.

  python rawpng.py encode in.raw W H out.png
"""
import struct
import sys
import zlib


def write_png(path, width, height, rows_rgb):
    """rows_rgb: list of bytearray, each width*3 bytes, RGB order."""
    raw = bytearray()
    for row in rows_rgb:
        raw.append(0)          # filter type 0 (None)
        raw.extend(row)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit truecolour
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
        f.write(chunk(b"IEND", b""))


def encode(raw_path, width, height, png_path):
    """ARGB8 .raw (PF_Pixel8 = A,R,G,B) -> RGB PNG."""
    with open(raw_path, "rb") as f:
        buf = f.read()
    need = width * height * 4
    if len(buf) != need:
        raise SystemExit(f"{raw_path}: {len(buf)} bytes, expected {need}")
    rows = []
    for y in range(height):
        base = y * width * 4
        row = bytearray(width * 3)
        for x in range(width):
            p = base + x * 4
            row[x * 3 + 0] = buf[p + 1]
            row[x * 3 + 1] = buf[p + 2]
            row[x * 3 + 2] = buf[p + 3]
        rows.append(row)
    write_png(png_path, width, height, rows)
    print("wrote", png_path)


if __name__ == "__main__":
    if sys.argv[1] != "encode":
        raise SystemExit("usage: rawpng.py encode in.raw W H out.png")
    encode(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
