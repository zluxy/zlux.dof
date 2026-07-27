import numpy as np
np.random.seed(3)
W,H=400,300
# Textured midtone surface (so banding would show as steps in the blur)
photo=(np.random.rand(H,W,3)*0.25+0.45)
depth=np.tile(np.linspace(0,1,W),(H,1))   # smooth full-range gradient
def to_argb(rgb):
    a=np.empty((H,W,4),np.uint8);a[...,0]=255
    a[...,1]=(np.clip(rgb[...,0],0,1)*255).astype(np.uint8);a[...,2]=(np.clip(rgb[...,1],0,1)*255).astype(np.uint8);a[...,3]=(np.clip(rgb[...,2],0,1)*255).astype(np.uint8)
    return a
to_argb(photo).tofile("band_photo.raw")
to_argb(np.repeat(depth[...,None],3,2)).tofile("band_depth.raw")
print("ok")
