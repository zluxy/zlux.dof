W,H=720,540
import sys
a=open(sys.argv[1],"rb").read(); b=open(sys.argv[2],"rb").read()
n=W*H; d=[]; big=0
for i in range(n):
    for c in (1,2,3):
        v=abs(a[i*4+c]-b[i*4+c]); d.append(v)
        if v>4: big+=1
d.sort()
print("%s vs %s  mean %.4f  median %d  p99.9 %d  max %d   (>4/255: %d of %d = %.4f%%)"
      %(sys.argv[1],sys.argv[2],sum(d)/len(d), d[len(d)//2], d[int(len(d)*0.999)], d[-1], big, len(d), 100.0*big/len(d)))
