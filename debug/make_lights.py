import numpy as np

W, H = 480, 360
# Photo: black bg with overlapping bright disc point-lights, different colors.
# Depth: each light at a different depth; background far. White=near convention.
photo = np.zeros((H, W, 3), dtype=np.float64)
depth = np.zeros((H, W), dtype=np.float64)  # 0=far(black) .. 1=near(white)

# Far flat background depth = 0.05 (far). Focus will be set near 0.0 so everything
# in the bg is defocused; lights at varying depths -> varying disc sizes & ordering.
depth[:] = 0.05

def stamp(cx, cy, d, col, r=3):
    yy, xx = np.ogrid[:H, :W]
    m = (xx-cx)**2 + (yy-cy)**2 <= r*r
    for c in range(3):
        photo[..., c][m] = col[c]
    depth[m] = d

# Cluster of overlapping lights at the SAME screen area but DIFFERENT depths,
# so their bokeh discs overlap and depth-ordering matters.
# d larger = nearer (whiter). Place a near red, a mid green, a far blue, tightly.
stamp(210, 180, 0.55, (1.0, 0.15, 0.15), r=3)   # near red  (should be ON TOP)
stamp(250, 180, 0.35, (0.15, 1.0, 0.15), r=3)   # mid green
stamp(230, 210, 0.18, (0.2, 0.4, 1.0),  r=3)    # far blue  (behind)

# A second pair far apart in depth but overlapping discs.
stamp(120, 130, 0.50, (1.0, 0.9, 0.3), r=3)     # near warm
stamp(150, 150, 0.12, (0.3, 0.6, 1.0), r=3)     # far cool

# A row of equal far lights (continuity / smooth check) at the bottom.
for i, x in enumerate(range(60, 440, 40)):
    stamp(x, 300, 0.10, (1.0, 1.0, 1.0), r=2)

photo = np.clip(photo, 0, 1)

def to_argb(rgb01):
    a = np.empty((H, W, 4), dtype=np.uint8)
    a[..., 0] = 255
    a[..., 1] = (rgb01[..., 0]*255).astype(np.uint8)
    a[..., 2] = (rgb01[..., 1]*255).astype(np.uint8)
    a[..., 3] = (rgb01[..., 2]*255).astype(np.uint8)
    return a

to_argb(photo).tofile("lights_photo.raw")
dep3 = np.repeat(depth[..., None], 3, axis=2)
to_argb(dep3).tofile("lights_depth.raw")
print("WROTE", W, H)
