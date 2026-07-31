import sys, struct
from PIL import Image
import numpy as np

# Downscale factor so iteration is fast (4608 -> 1152 tall).
SCALE = 4

def dump(path_png, path_raw):
    im = Image.open(path_png).convert("RGB")
    w, h = im.size
    w2, h2 = w // SCALE, h // SCALE
    im = im.resize((w2, h2), Image.LANCZOS)
    rgb = np.asarray(im, dtype=np.uint8)            # (h, w, 3) RGB
    argb = np.empty((h2, w2, 4), dtype=np.uint8)    # PF_Pixel8 = A,R,G,B
    argb[..., 0] = 255
    argb[..., 1] = rgb[..., 0]
    argb[..., 2] = rgb[..., 1]
    argb[..., 3] = rgb[..., 2]
    argb.tofile(path_raw)
    print(f"{path_png}: {w}x{h} -> {w2}x{h2} -> {path_raw}")
    return w2, h2

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dump":
        w, h = dump(sys.argv[2], sys.argv[3])
        print(f"DIMS {w} {h}")
    elif cmd == "encode":
        # encode raw ARGB8 -> png
        path_raw, w, h, path_png = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        a = np.fromfile(path_raw, dtype=np.uint8).reshape((h, w, 4))
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = a[..., 1]
        rgb[..., 1] = a[..., 2]
        rgb[..., 2] = a[..., 3]
        Image.fromarray(rgb, "RGB").save(path_png)
        print("wrote", path_png)
