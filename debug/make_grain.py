import numpy as np
np.random.seed(7)
W,H=400,300
photo=np.zeros((H,W,3),np.float64)
depth=np.full((H,W),0.06,np.float64)        # all far
depth[0:3,0:3]=0.0; depth[0:3,W-3:W]=1.0     # anchors
# Dense random bright specks across the whole frame (distant foliage highlights).
ys=np.random.randint(0,H,1400); xs=np.random.randint(0,W,1400)
for x,y in zip(xs,ys):
    c=np.random.rand()*0.7+0.3
    photo[y,x]=(c,c*0.95,c*0.8)
def to_argb(rgb):
    a=np.empty((H,W,4),np.uint8);a[...,0]=255
    a[...,1]=(np.clip(rgb[...,0],0,1)*255).astype(np.uint8);a[...,2]=(np.clip(rgb[...,1],0,1)*255).astype(np.uint8);a[...,3]=(np.clip(rgb[...,2],0,1)*255).astype(np.uint8)
    return a
to_argb(photo).tofile("gn_photo.raw")
to_argb(np.repeat(depth[...,None],3,2)).tofile("gn_depth.raw")
print("ok")
