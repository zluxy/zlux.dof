import numpy as np
W,H=400,300
photo=np.zeros((H,W,3),np.float64)
depth=np.empty((H,W),np.float64)
depth[:, :150]=0.0; depth[:,250:]=1.0; depth[:,150:250]=0.06
# Bright YELLOW far field fills the strip (far-most), photo:
photo[:,150:250]=(1.0,0.8,0.1)
# A bright RED vertical bar, NEARER far depth, crossing the field.
bx0,bx1=192,208
photo[:,bx0:bx1]=(1.0,0.1,0.1)
depth[:,bx0:bx1]=0.30      # nearer than the 0.06 field -> should occlude it
def to_argb(rgb):
    a=np.empty((H,W,4),np.uint8);a[...,0]=255
    a[...,1]=(rgb[...,0]*255).astype(np.uint8);a[...,2]=(rgb[...,1]*255).astype(np.uint8);a[...,3]=(rgb[...,2]*255).astype(np.uint8)
    return a
to_argb(np.clip(photo,0,1)).tofile("l9_photo.raw")
to_argb(np.repeat(depth[...,None],3,2)).tofile("l9_depth.raw")
print("ok")
