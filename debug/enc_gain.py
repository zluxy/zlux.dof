import sys, numpy as np
from PIL import Image
raw,w,h,out,g = sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4],float(sys.argv[5])
a=np.fromfile(raw,np.uint8).reshape((h,w,4)).astype(np.float64)
rgb=np.stack([a[...,1],a[...,2],a[...,3]],-1)*g
Image.fromarray(np.clip(rgb,0,255).astype(np.uint8),"RGB").save(out)
print("wrote",out)
