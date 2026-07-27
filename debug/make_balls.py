import numpy as np
np.random.seed(11)
W,H=420,320
photo=np.zeros((H,W,3),np.float64)
depth=np.empty((H,W),np.float64)
depth[:,:120]=0.0; depth[:,300:]=1.0; depth[:,120:300]=0.5   # dense anchors + mid
# Many bright coloured discs at DIFFERENT far depths, overlapping heavily.
pal=[(1,0.2,0.2),(0.2,1,0.3),(0.3,0.5,1),(1,0.9,0.3),(1,0.4,1),(0.3,1,1)]
for k in range(22):
    cx=np.random.randint(150,290); cy=np.random.randint(70,250)
    d=np.random.uniform(0.05,0.42)     # all FAR (< focus 0.6), varied depths
    col=pal[k%len(pal)]; r=np.random.randint(5,9)
    yy,xx=np.ogrid[:H,:W]; m=(xx-cx)**2+(yy-cy)**2<=r*r
    for c in range(3): photo[...,c][m]=col[c]
    depth[m]=d
photo=np.clip(photo,0,1)
def to_argb(rgb):
    a=np.empty((H,W,4),np.uint8);a[...,0]=255
    a[...,1]=(rgb[...,0]*255).astype(np.uint8);a[...,2]=(rgb[...,1]*255).astype(np.uint8);a[...,3]=(rgb[...,2]*255).astype(np.uint8)
    return a
to_argb(photo).tofile("balls_photo.raw")
to_argb(np.repeat(depth[...,None],3,2)).tofile("balls_depth.raw")
print("ok")
