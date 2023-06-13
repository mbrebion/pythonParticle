import matplotlib.pyplot as plt
import numpy as np

plt.grid()
plt.xlabel("x/x0")
plt.ylabel("E_c * x / (E_c *x)_0")
# plt.ylim(0.8,1.25)
plt.ylim(0.98, 1.08)


def fun(ec, x, corr):
    return ec * (x * 0.01 - corr)


def addPlot(name, display, corr):
    f = open(name, "r")
    ts = []
    xs = []
    Ecs = []
    Ss = []
    for l in f:
        t, x, Ec, Ecr, r = l.split()
        ts.append(float(t))
        xs.append(float(x))
        Ecs.append(float(Ec))
        Ss.append(1 * float(x))
    Ecs = np.array(Ecs)
    xs = np.array(xs)
    init = fun(Ecs[0], xs[0], corr)
    ys = fun(Ecs, xs, corr) / init
    xs = (xs * 0.01 - corr) / (xs[0] * 0.01 - corr)

    plt.plot(xs, ys, display, label=name)
    a, b, c = np.polyfit(xs-1,ys,2)
    #plt.plot(xs, + a*(xs-1)**2 + b*(xs-1)+c)
    return a,b,c

invNs = []
As = []
Bs = []
a,b,c = addPlot("dc200mum256.txt", "--g", 0.000000766914*0.5*2**0.5)

a,b,c = addPlot("dc100mum128k.txt", "-.y", 0.0000030687*2**0.5)
a,b,c = addPlot("dc200mum128k.txt", "--y", 0.000000766914*1*2**0.5)
As.append(a)
Bs.append(b)
invNs.append(1/128.)


a,b,c = addPlot("dc200mum64k.txt", "--r", 0.00000153160*2**0.5)
As.append(a)
Bs.append(b)
invNs.append(1/64.)
# addPlot("dc200mum64k.txt", "-.r", 0.00000153160 * 1)

a,b,c = addPlot("dc200mum32k.txt", "--k", 0.00000306765*2**0.5)
As.append(a)
Bs.append(b)
invNs.append(1/32.)

# addPlot("dc200mum32k.txt", "-.k", )

a,b,c = addPlot("dc200mum16k.txt", "--b",  0.0000061359*2**0.5)
As.append(a)
Bs.append(b)
invNs.append(1/16.)
# addPlot("dc200mum16k.txt", "-.b", 0.0000061359)


As = np.array(As)
Bs = np.array(Bs)
invNs = np.array(invNs)

B,lnbeta = np.polyfit(np.log(invNs),np.log(-Bs),1)
A,lnalpha = np.polyfit(np.log(invNs),np.log(As),1)

print(As)
print(A)

print(Bs)
print(B)


plt.legend()
plt.show()
