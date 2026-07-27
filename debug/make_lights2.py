import numpy as np
W, H = 400, 300
photo = np.zeros((H, W, 3), dtype=np.float64)
depth = np.full((H, W), 0.05, dtype=np.float64)  # far flat bg
def stamp(cx, cy, d, col, r):
    yy, xx = np.ogrid[:H, :W]
    m = (xx-cx)**2 + (yy-cy)**2 <= r*r
    for c in range(3): photo[..., c][m] = col[c]
    depth[m] = d
# Two bright lights, big, heavily overlapping, very different depth.
stamp(175, 150, 0.80, (1.0, 0.1, 0.1), 9)   # NEAR red  (closer to focus=0.9 -> on top)
stamp(225, 150, 0.20, (0.1, 0.8, 1.0), 9)   # FAR cyan  (far -> behind)
photo = np.clip(photo, 0, 1)
def to_argb(rgb):
    a = np.empty((H, W, 4), np.uint8); a[...,0]=255
    a[...,1]=(rgb[...,0]*255).astype(np.uint8); a[...,2]=(rgb[...,1]*255).astype(np.uint8); a[...,3]=(rgb[...,2]*255).astype(np.uint8)
    return a
to_argb(photo).tofile("l2_photo.raw")
to_argb(np.repeat(depth[...,None],3,2)).tofile("l2_depth.raw")
print("ok")
