W,H=1280,720
X0,X1,Y0,Y1=700,1200,100,600
def stats(f):
    buf=open(f,"rb").read()
    lum=[[ (buf[(y*W+x)*4+1]*0.299+buf[(y*W+x)*4+2]*0.587+buf[(y*W+x)*4+3]*0.114)
           for x in range(X0,X1)] for y in range(Y0,Y1)]
    h=len(lum); w=len(lum[0]); dev=[]
    for y in range(2,h-2):
        for x in range(2,w-2):
            m=sum(lum[y+dy][x+dx] for dy in(-2,0,2) for dx in(-2,0,2))/9.0
            dev.append(abs(lum[y][x]-m))
    dev.sort()
    return sum(sum(r) for r in lum)/(w*h), dev[int(len(dev)*0.999)], dev[-1]
print("boost  weight-LOD   mean   p99.9   max    <- speckle in the noise field, bdef=1.0")
for b in ["1.0","3.0"]:
    for tag in ["on","off"]:
        m,p,mx=stats("wl_%s_%s.raw"%(tag,b))
        print("%-6s %-11s %5.1f  %6.2f  %6.2f"%(b,("split "+tag),m,p,mx))
