import numpy as np

data=[]
f = open("data.txt", "r")
for l in f:
    d = float(l)
    if d > 25:
        print(d)
    else:
        data.append(abs(float(l)))

dist = np.array(data)
print(np.mean(dist), np.std(dist) , np.std(dist) / len(dist)**0.5)