import sys
sys.path.append("../../")
from pyttn import *


def build_topology_from_string_multiset(N1, N2, N3, N4, N5, Nb, m):
    ret = "(1(N1(N2(Nb(m))(Nb(m)))(N2(Nb(m))(Nb(m))(Nb(m))))(N1(N3(N4(Nb(m))(Nb(m))(Nb(m)))(N4(Nb(m))(Nb(m))(Nb(m)))(N4(Nb(m))(Nb(m))(Nb(m))))(N3(N5(Nb(m))(Nb(m)))(N5(Nb(m))(Nb(m))(Nb(m))(Nb(m)))(N5(Nb(m))(Nb(m))(Nb(m))(Nb(m))))))"
    ret=ret.replace("N1", str(N1))
    ret=ret.replace("N2", str(N2))
    ret=ret.replace("N3", str(N3))
    ret=ret.replace("N4", str(N4))
    ret=ret.replace("N5", str(N5))
    ret=ret.replace("Nb", str(Nb))
    ret=ret.replace("m", str(m))
    res = ntree(ret)
    ntreeBuilder.sanitise(res)
    return res


def build_topology_from_string(N1, N2, N3, N4, N5, Nb, m):
    ret = "(1(2(2))(2(N1(N2(Nb(m))(Nb(m)))(N2(Nb(m))(Nb(m))(Nb(m))))(N1(N3(N4(Nb(m))(Nb(m))(Nb(m)))(N4(Nb(m))(Nb(m))(Nb(m)))(N4(Nb(m))(Nb(m))(Nb(m))))(N3(N5(Nb(m))(Nb(m)))(N5(Nb(m))(Nb(m))(Nb(m))(Nb(m)))(N5(Nb(m))(Nb(m))(Nb(m))(Nb(m)))))))"
    ret=ret.replace("N1", str(N1))
    ret=ret.replace("N2", str(N2))
    ret=ret.replace("N3", str(N3))
    ret=ret.replace("N4", str(N4))
    ret=ret.replace("N5", str(N5))
    ret=ret.replace("Nb", str(Nb))
    ret=ret.replace("m", str(m))
    return ntree(ret)

def build_topology(N1, N2, N3, N4, N5, Nb, m):
    topo = ntree()
    topo.insert(1)
    #add electronic degrees of freedom
    topo().insert(2)
    topo()[0].insert(2)
    topo().insert(2)
 
    topo()[1].insert(N1)

    topo()[1][0].insert(N2)
 
    topo()[1][0][0].insert(Nb)
    topo()[1][0][0][0].insert(m[0])
    topo()[1][0][0].insert(Nb)
    topo()[1][0][0][1].insert(m[1])


    topo()[1][0].insert(N2)
 
    topo()[1][0][1].insert(Nb)
    topo()[1][0][1][0].insert(m[2])
    topo()[1][0][1].insert(Nb)
    topo()[1][0][1][1].insert(m[3])
    topo()[1][0][1].insert(Nb)
    topo()[1][0][1][2].insert(m[4])

    topo()[1].insert(N1)
 
    topo()[1][1].insert(N3)
 
    topo()[1][1][0].insert(N4)
 
    topo()[1][1][0][0].insert(Nb)
    topo()[1][1][0][0][0].insert(m[5])
    topo()[1][1][0][0].insert(Nb)
    topo()[1][1][0][0][1].insert(m[6])
    topo()[1][1][0][0].insert(Nb)
    topo()[1][1][0][0][2].insert(m[7])
            
    topo()[1][1][0].insert(N4)
 
    topo()[1][1][0][1].insert(Nb)
    topo()[1][1][0][1][0].insert(m[8])
    topo()[1][1][0][1].insert(Nb)
    topo()[1][1][0][1][1].insert(m[9])
    topo()[1][1][0][1].insert(Nb)
    topo()[1][1][0][1][2].insert(m[10])

    topo()[1][1][0].insert(N4)
 
    topo()[1][1][0][2].insert(Nb)
    topo()[1][1][0][2][0].insert(m[11])
    topo()[1][1][0][2].insert(Nb)
    topo()[1][1][0][2][1].insert(m[12])
    topo()[1][1][0][2].insert(Nb)
    topo()[1][1][0][2][2].insert(m[13])

    topo()[1][1].insert(N3)
 
    topo()[1][1][1].insert(N5)
 
    topo()[1][1][1][0].insert(Nb)
    topo()[1][1][1][0][0].insert(m[14])
    topo()[1][1][1][0].insert(Nb)
    topo()[1][1][1][0][1].insert(m[15])

    topo()[1][1][1].insert(N5)
 
    topo()[1][1][1][1].insert(Nb)
    topo()[1][1][1][1][0].insert(m[16])
    topo()[1][1][1][1].insert(Nb)
    topo()[1][1][1][1][1].insert(m[17])
    topo()[1][1][1][1].insert(Nb)
    topo()[1][1][1][1][2].insert(m[18])
    topo()[1][1][1][1].insert(Nb)
    topo()[1][1][1][1][3].insert(m[19])

    topo()[1][1][1].insert(N5)
 
    topo()[1][1][1][2].insert(Nb)
    topo()[1][1][1][2][0].insert(m[20])
    topo()[1][1][1][2].insert(Nb)
    topo()[1][1][1][2][1].insert(m[21])
    topo()[1][1][1][2].insert(Nb)
    topo()[1][1][1][2][2].insert(m[22])
    topo()[1][1][1][2].insert(Nb)
    topo()[1][1][1][2][3].insert(m[23])
    return topo
