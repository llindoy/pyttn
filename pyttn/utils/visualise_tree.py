from pyttn import ntree


def visualise_tree(tree, prog="dot", ax=None, node_size=500, linewidth=3, add_labels=True):
    """A function for plotting an ntree object

    :param tree: The tree to be visualised
    :type tree: ntree
    :param prog: The visualisation function used with graphviz_layout (default: "dot")
    :type prog: str, optional
    :param ax: Draw the tree in specified Matplotlib axes
    :type prog: Matplotlib Axes object, optional
    :param node_size: The node size used for visualisation (default: 500)
    :type node_size: int, optional
    :param linewidth: The width of edges connecting nodes (default: 3)
    :type linewidth: int, optional
    :param add_labels: Whether or not to include the node labels (default: True)
    :type add_labels: bool, optional

    """
    try:
        import networkx as nx
        from networkx.drawing.nx_pydot import graphviz_layout

        G = nx.Graph()

        # add all of the nodes to the tree
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
        nx.draw(G, pos, labels=labeldict,
                node_size=node_size, width=linewidth, ax=None)
    except Exception as e:
        print(e)
        raise RuntimeError("Failed to visualise tree structure.")
