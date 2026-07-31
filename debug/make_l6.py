import numpy as np
W,H=400,300
photo=np.zeros((H,W,3),np.float64)
depth=np.empty((H,W),np.float64)
# Dense [0,1] anchors so auto-range stays identity: left far(0), right near(1).
depth[:, :160]=0.0
depth[:, 240:]=1.0
depth[:,160:240]=0.15            # mid strip, FAR
def stamp(cx,cy,d,col,r=7):
    yy,xx=np.ogrid[:H,:W]; m=(xx-cx)**2+(yy-cy)**2<=r*r
    for c in range(3): photo[...,c][m]=col[c]
    depth[m]=d
# Two FAR lights in the strip, distinct far depths, overlapping discs.
stamp(190,150,0.15,(1.0,0.1,0.1),7)   # A red  d->0.85 (farther, behind)
stamp(210,150,0.30,(0.1,0.7,1.0),7)   # B cyan d->0.70 (nearer, ON TOP)
photo=np.clip(photo,0,1)
def to_argb(rgb):
    a=np.empty((H,W,4),np.uint8);a[...,0]=255
    a[...,1]=(rgb[...,0]*255).astype(np.uint8);a[...,2]=(rgb[...,1]*255).astype(np.uint8);a[...,3]=(rgb[...,2]*255).astype(np.uint8)
    return a
to_argb(photo).tofile("l6_photo.raw")
to_argb(np.repeat(depth[...,None],3,2)).tofile("l6_depth.raw")
print("ok")
