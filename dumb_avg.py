import numpy as np

data = np.zeros((100, 7))

sum = 0
for i in range(5000, 6000):
    data[:,:4] += np.loadtxt(f"../STOTools/example/OP/OP_T40_p80/op{i}.out")[:,:4]
    data[:,4:] += np.loadtxt(f"../STOTools/example/POL/POL_T40_p80/pol{i}.out")[:,1:4]
    sum += 1
data /= sum
np.savetxt("dumb.out", data)
