import os
os.environ["OMP_NUM_THREADS"] = "1"

from pyttn import ntree, ntreeBuilder


def get_nskip(Nl, d=3):
    ret = 0
    for i in range(Nl):
        if i == 0:
            ret += 1
        else:
            ret += d * (d - 1) ** (i - 1)
    return ret


def get_index(ind, d=3):
    if len(ind) == 0:
        return 0
    else:
        c = get_nskip(len(ind))
        ns = [2**i for i in reversed(range(len(ind)))]
        ls = [d - 1 for i in range(len(ind))]
        ls[0] = d

        for i in range(len(ind)):
            c += ind[i] * ns[i]
        return c

def get_spin_connectivity(Nl, d=3):
    topo = ntree("(1)")

    inds = []

    # build the cayley tree.
    if Nl > 0:

        # add the first layer
        x = get_index([])
        for i in range(d):
            y = get_index([i])
            inds.append([x, y])
            topo().insert(1)

        for layer in range(1, Nl):
            # get all leaves
            leaf_indices = topo.leaf_indices()

            for li in leaf_indices:
                x = get_index(li)
                for i in range(d - 1):
                    topo.at(li).insert(1)
                    y = get_index(li + [i])
                    inds.append([x, y])

    print(inds)
    return inds, topo.size()


def build_system_topology(Nl, chi, d=3):
    topo = ntree('(1)')

    #now build each of the Layers around the tree
    if Nl > 0:
        #add the first layer
        for i in range(d):
            topo().insert(chi)

        #now attempt to add all other layers to the tree
        for layer in range(1, Nl):
            #get all of the leaves of the the tree with the current number of layers
            leaf_indices = topo.leaf_indices()

            #now iterate over each leaf and add its children
            for li in leaf_indices:
                for i in range(d-1):
                    topo.at(li).insert(chi)
                    topo.at

    return topo


def build_topology(Nl, ds, chi, chiS, chiB, nbose, discbath, degree, d=3):
    topo = build_system_topology(Nl, chi, d=d)
    print(topo)

    # now get each site using a depth first search traversal and determine the indices associated with the site
    indices = []
    for i in topo.dfs():
        indices.append(i.index())

    # now iterating backwards through each sites.  Attach the spin and bath degrees of freedom as the first child of this site.
    # by performing this process in reverse order we ensure we do not affect the ordering of progressive sites
    for ni in reversed(indices):
        topo.at(ni).insert_front(chiS)
        topo.at(ni)[0].insert(ds)
        #now if the discrete bath object has been created we add the bath modes to the tre
        if discbath is not None:
            _ = discbath.add_bath_tree(topo.at(ni)[0], degree, chiB, min(chiB, nbose))

    #ntreeBuilder.sanitise(topo)

    return topo

def get_mode_reordering(Nl, Nb, d=3, site_size=1):
    #first get the system tree.  
    topo = build_system_topology(Nl, 1, d=d)
    print(topo)

    #traverse the nodes in depth first search order and create a dictionary storing the node and the index the order in which it was reach
    dict_dfs = {}
    c=0
    for  i in topo.dfs():
        dict_dfs[i] = c
        c=c+1

    print(len(dict_dfs))

    ordering = []
    #now iterate over it in bfs ordering and for the ith leaf in bfs order set it to point to the index in dfs order.
    #Additionally we set all the bath modes to be ordered correctly
    for i in topo.bfs():
        ordering += [dict_dfs[i]*(Nb+site_size) + j for j in range(Nb+site_size)]
    print(ordering)
    return ordering