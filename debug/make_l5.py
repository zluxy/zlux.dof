import numpy as np
W,H=400,300
photo=np.zeros((H,W,3),np.float64)
depth=np.full((H,W),0.08,np.float64)        # bg far
depth[0:3,0:3]=0.0; depth[0:3,W-3:W]=1.0     # anchors so auto-range ~ identity
def stamp(cx,cy,d,col,r=7):
    yy,xx=np.ogrid[:H,:W]; m=(xx-cx)**2+(yy-cy)**2<=r*r
    for c in range(3): photo[...,c][m]=col[c]
    depth[m]=d
# Both FAR (depth < focus 0.6). A farther (lower depth) -> behind; B nearer -> on top.
stamp(186,150,0.15,(1.0,0.1,0.1),7)   # A red  (farther, behind)
stamp(212,150,0.35,(0.1,0.7,1.0),7)   # B cyan (nearer, ON TOP)
photo=np.clip(photo,0,1)
def to_argb(rgb):
    a=np.empty((H,W,4),np.uint8);a[...,0]=255
    a[...,1]=(rgb[...,0]*255).astype(np.uint8);a[...,2]=(rgb[...,1]*255).astype(np.uint8);a[...,3]=(rgb[...,2]*255).astype(np.uint8)
    return a
to_argb(photo).tofile("l5_photo.raw")
to_argb(np.repeat(depth[...,None],3,2)).tofile("l5_depth.raw")
print("ok")
