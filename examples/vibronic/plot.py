import numpy as np
import matplotlib.pyplot as plt
import h5py

h5 = h5py.File("res.h5", "r")
t = np.array(h5.get("t"))
res = np.array(h5.get("res"))
h5.close()

print(res)
plt.plot(t, res[0, :])
plt.show()