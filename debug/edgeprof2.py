W,H,CY=1280,720,570
def prof(f):
    buf=open(f,"rb").read(); out=[]
    for x in range(0,240):
        s=sum(buf[((CY+dy)*W+x)*4+1] for dy in range(-2,3))
        out.append(s/5.0)
    return out
ps={d:prof("bq_%s.raw"%d) for d in ["0.0","0.35","0.65","1.0"]}
for d,p in ps.items():
    pk=max(p); hi=lo=None
    for x in range(p.index(pk),len(p)):
        if hi is None and p[x]<=0.9*pk: hi=x
        if lo is None and p[x]<=0.1*pk: lo=x; break
    print("bdef %-5s peak %6.1f   10-90%% edge %s px" % (d,pk,(lo-hi) if lo and hi else "?"))
print(); print("x     "+"  ".join("%6s"%d for d in ps))
for x in range(40,180,6):
    print("%-5d "%x+"  ".join("%6.1f"%ps[d][x] for d in ps))
