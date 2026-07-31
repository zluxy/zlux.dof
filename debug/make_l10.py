import numpy as np
W,H=400,300
photo=np.zeros((H,W,3),np.float64)
depth=np.full((H,W),0.5,np.float64)          # background mid-depth (will be in focus)
depth[0:3,0:3]=0.0; depth[0:3,W-3:W]=1.0      # anchors
# Background: bold vertical colour stripes so a reveal is unmistakable.
cols=[(1,0.15,0.15),(0.15,1,0.15),(0.2,0.4,1),(1,1,0.3),(1,1,1)]
for x in range(W):
    photo[:,x]=cols[(x//28)%len(cols)]
# Foreground: solid GREY vertical bar, NEAR depth (white), centred -> defocuses.
bx0,bx1=175,225
photo[:,bx0:bx1]=(0.5,0.5,0.5)
depth[:,bx0:bx1]=0.95
def to_argb(rgb):
    a=np.empty((H,W,4),np.uint8);a[...,0]=255
    a[...,1]=(np.clip(rgb[...,0],0,1)*255).astype(np.uint8);a[...,2]=(np.clip(rgb[...,1],0,1)*255).astype(np.uint8);a[...,3]=(np.clip(rgb[...,2],0,1)*255).astype(np.uint8)
    return a
to_argb(photo).tofile("l10_photo.raw")
to_argb(np.repeat(depth[...,None],3,2)).tofile("l10_depth.raw")
print("ok")
