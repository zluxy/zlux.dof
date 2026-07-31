W,H=1280,720
X0,X1,Y0,Y1=700,1200,100,600
def stats(f):
    buf=open(f,"rb").read()
    lum=[[ (buf[(y*W+x)*4+1]*0.299+buf[(y*W+x)*4+2]*0.587+buf[(y*W+x)*4+3]*0.114)
           for x in range(X0,X1)] for y in range(Y0,Y1)]
    h=len(lum); w=len(lum[0])
    dev=[]
    for y in range(2,h-2):
        for x in range(2,w-2):
            m=sum(lum[y+dy][x+dx] for dy in(-2,0,2) for dx in(-2,0,2))/9.0
            dev.append(abs(lum[y][x]-m))
    dev.sort()
    mean=sum(lum[y][x] for y in range(h) for x in range(w))/(w*h)
    return mean, dev[len(dev)//2], dev[int(len(dev)*0.999)], dev[-1]
print("bdef   mean   med|dev|  p99.9  max   <- local high-freq deviation (speckle)")
for d in ["0.0","0.35","0.65","1.0"]:
    m,md,p,mx=stats("bq_%s.raw"%d)
    print("%-6s %5.1f   %6.2f  %5.1f  %5.1f"%(d,m,md,p,mx))
