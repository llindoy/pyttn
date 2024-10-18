from pyttn import ntree

def visualise_tree(tree, prog="dot", node_size=500, linewidth=3, add_labels=True):
    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout

    G = nx.Graph()

    #add all of the nodes to the tree
    labeldict = {}
    for i in tree:
        G.add_node(i)
        if add_labels:
            labeldict[i] = str(i.data)

    for i in tree:
        c = 0
        if not i.is_leaf():
            for ind in range(i.size()):
                G.add_edge(i, i[ind])

    pos = graphviz_layout(G, prog=prog, root=tree())
    nx.draw(G, pos, labels=labeldict, node_size=node_size, width=linewidth)
