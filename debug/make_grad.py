import numpy as np
W,H=400,300
photo=np.zeros((H,W,3),np.float64)
depth=np.tile(np.linspace(0,1,W),(H,1))  # left=0 black(far?), right=1 white(near?)
def stamp(cx,cy,col,r=4):
    yy,xx=np.ogrid[:H,:W]; m=(xx-cx)**2+(yy-cy)**2<=r*r
    for c in range(3): photo[...,c][m]=col[c]
for x in range(40,400,60):
    stamp(x,150,(1,1,1),4)
def to_argb(rgb):
    a=np.empty((H,W,4),np.uint8);a[...,0]=255
    a[...,1]=(rgb[...,0]*255).astype(np.uint8);a[...,2]=(rgb[...,1]*255).astype(np.uint8);a[...,3]=(rgb[...,2]*255).astype(np.uint8)
    return a
to_argb(photo).tofile("grad_photo.raw")
to_argb(np.repeat(depth[...,None],3,2)).tofile("grad_depth.raw")
print("ok")
