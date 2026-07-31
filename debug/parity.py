W,H=720,540
a=open("p31_cpu.raw","rb").read(); b=open("p31_gpu.raw","rb").read()
n=W*H; d=[]; big=0
for i in range(n):
    for c in (1,2,3):
        v=abs(a[i*4+c]-b[i*4+c]); d.append(v)
        if v>4: big+=1
d.sort()
print("CPU vs GPU  mean %.3f  median %d  p99.9 %d  max %d   (>4/255: %d of %d = %.4f%%)"
      %(sum(d)/len(d), d[len(d)//2], d[int(len(d)*0.999)], d[-1], big, len(d), 100.0*big/len(d)))
