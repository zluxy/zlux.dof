import numpy as np
W,H=400,300
photo=np.zeros((H,W,3),np.float64)
depth=np.empty((H,W),np.float64)
depth[:, :150]=0.0; depth[:,250:]=1.0; depth[:,150:250]=0.10   # dense anchors + far strip
def stamp(cx,cy,d,col,r):
    yy,xx=np.ogrid[:H,:W]; m=(xx-cx)**2+(yy-cy)**2<=r*r
    for c in range(3): photo[...,c][m]=col[c]
    depth[m]=d
# Two BIG bright discs, very different far depths, ~60% overlap.
stamp(185,150,0.10,(1.0,0.85,0.2),34)   # A warm  d->0.90 far (very blurred, behind)
stamp(216,150,0.34,(0.2,0.6,1.0),34)    # B blue  d->0.66 far (less blur, ON TOP)
photo=np.clip(photo,0,1)
def to_argb(rgb):
    a=np.empty((H,W,4),np.uint8);a[...,0]=255
    a[...,1]=(rgb[...,0]*255).astype(np.uint8);a[...,2]=(rgb[...,1]*255).astype(np.uint8);a[...,3]=(rgb[...,2]*255).astype(np.uint8)
    return a
to_argb(photo).tofile("l7_photo.raw")
to_argb(np.repeat(depth[...,None],3,2)).tofile("l7_depth.raw")
print("ok")
