import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("../../")
from pyttn import *

from sbm_core import sbm_discretise

msop = multiset_SOP(10, 2)
msop[1,1] += 0.1
print(msop[1,1])

