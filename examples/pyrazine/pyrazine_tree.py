def build_topology_mode_combination(N1, N2, N3, N4, N5, m):
    from pyttn import ntree, ntreeBuilder

    """ Construct the pyrazine ML-MCTDH tree structure used in O. Vendrell and H.-D. Meyer, J. Chem. Phys. 134, 044135 (2011).
    This structure makes use of mode combination to treat a set of 8 composite vibrational modes rather than the full 24 modes 
    in the full tree structure.   

    :param N1: Bond dimension N1
    :type N1: int
    :param N2: Bond dimension N2
    :type N2: int
    :param N3: Bond dimension N3
    :type N3: int
    :param N4: Bond dimension N4
    :type N4: int
    :param N5: Bond dimension N5
    :type N5: int
    :param m: A list containing the 8 local Hilbert space dimensions of the composite vibrational modes
    :type m: list[int]
    """
    topo = ntree()
    topo.insert(1)
    # add electronic degrees of freedom
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
    ntreeBuilder.sanitise(topo)
    return topo
