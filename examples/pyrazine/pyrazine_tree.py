import sys
sys.path.append("../../")
from pyttn import *

#build the tree structure 
def build_topology_mode_combination(N1, N2, N3, N4, N5, m):
    topo = ntree()
    topo.insert(1)
    #add electronic degrees of freedom
    topo().insert(2)
    topo()[0].insert(2)
    topo().insert(2)
 
    topo()[1].insert(N1)

    topo()[1][0].insert(N2)
    topo()[1][0][0].insert(m[0])

    topo()[1][0].insert(N2)
 
    topo()[1][0][1].insert(m[1])

    topo()[1].insert(N1)
    topo()[1][1].insert(N3)
    topo()[1][1][0].insert(N4)
 
    topo()[1][1][0][0].insert(m[2])
            
    topo()[1][1][0].insert(N4)
 
    topo()[1][1][0][1].insert(m[3])

    topo()[1][1][0].insert(N4)
 
    topo()[1][1][0][2].insert(m[4])

    topo()[1][1].insert(N3)
    topo()[1][1][1].insert(N5)
 
    topo()[1][1][1][0].insert(m[5])

    topo()[1][1][1].insert(N5)
    topo()[1][1][1][1].insert(m[6])

    topo()[1][1][1].insert(N5)
    topo()[1][1][1][2].insert(m[7])
    return topo


