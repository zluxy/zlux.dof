import numpy as np
W, H = 400, 300
photo = np.zeros((H, W, 3), dtype=np.float64)
depth = np.full((H, W), 0.05, dtype=np.float64)
def stamp(cx, cy, d, col, r):
    yy, xx = np.ogrid[:H, :W]; m=(xx-cx)**2+(yy-cy)**2<=r*r
    for c in range(3): photo[...,c][m]=col[c]
    depth[m]=d
# Heavily overlapping: 22px apart, depths 0.72 / 0.40, focus will be 0.9 (both far)
stamp(189, 150, 0.72, (1.0, 0.1, 0.1), 7)   # near-ish red (smaller coc, ON TOP)
stamp(211, 150, 0.40, (0.1, 0.7, 1.0), 7)   # farther cyan (bigger coc, behind)
photo=np.clip(photo,0,1)
def to_argb(rgb):
    a=np.empty((H,W,4),np.uint8); a[...,0]=255
    a[...,1]=(rgb[...,0]*255).astype(np.uint8);a[...,2]=(rgb[...,1]*255).astype(np.uint8);a[...,3]=(rgb[...,2]*255).astype(np.uint8)
    return a
to_argb(photo).tofile("l3_photo.raw")
to_argb(np.repeat(depth[...,None],3,2)).tofile("l3_depth.raw")
print("ok")
