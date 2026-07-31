"""Bokeh Definition test scene (stdlib only -- see rawpng.py).

Left half : isolated point lights of a few sizes on black -- the disc-edge test.
            A crisp gather turns a 1px light into a hard-edged disc; a
            pre-filtered one turns it into a soft blob.
Right half: dense high-frequency noise with sparse blown pixels at the same
            depth -- the speckle test. This is the pattern the old footprint
            floor existed to hide, so it must NOT get worse.

Everything sits at depth 0.85 (far); render with focus 0.05 so the whole frame
is background and a single Far gather covers it.
"""
import random
import struct

W, H = 1280, 720   # the size the v3.1 changelog numbers were measured at

photo = [[(0.0, 0.0, 0.0)] * W for _ in range(H)]
depth = 0.85


def dot(cx, cy, r, col):
    r2 = r * r
    for y in range(max(0, cy - 4), min(H, cy + 5)):
        for x in range(max(0, cx - 4), min(W, cx + 5)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                photo[y][x] = col


# ── Left: point lights, radii 0.4 (single pixel) .. 3 px, a few colours.
cols = [(1, 1, 1), (1, .85, .5), (.5, .8, 1), (1, .4, .4)]
for i, r in enumerate([0.4, 1.0, 2.0, 3.0]):
    for j, col in enumerate(cols):
        dot(100 + j * 140, 90 + i * 160, r, col)

# ── Right: high-frequency texture with sparse fireflies.
rng = random.Random(7)
for y in range(H):
    for x in range(W // 2, W):
        if rng.random() > 0.995:
            photo[y][x] = (1.0, 1.0, 1.0)          # blown pixel
        else:
            photo[y][x] = tuple(rng.random() * 0.35 + 0.08 for _ in range(3))


def clamp8(v):
    i = int(v * 255.0 + 0.5)
    return 0 if i < 0 else (255 if i > 255 else i)


with open("bd_photo.raw", "wb") as f:
    for row in photo:
        f.write(bytes(b for c in row for b in (255, clamp8(c[0]), clamp8(c[1]), clamp8(c[2]))))

d8 = clamp8(depth)
with open("bd_depth.raw", "wb") as f:
    f.write(bytes([255, d8, d8, d8]) * (W * H))

print(f"ok {W}x{H}")
