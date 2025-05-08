# This files is part of the pyTTN package.
# (C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Optional

from pyttn.ttnpp import ntree
from pyttn.ttns.ttn.ttnExt import ttn_type, ms_ttn_type


def visualise_ntree(
    tree: ntree,
    prog: str = "dot",
    ax=None,
    node_size: int = 300,
    linewidth: int = 3,
    add_labels: bool = True,
) -> None:
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
            if not i.is_leaf():
                for ind in range(i.size()):
                    G.add_edge(i, i[ind])

        pos = graphviz_layout(G, prog=prog, root=tree())
        nx.draw(G, pos, labels=labeldict, node_size=node_size, width=linewidth, ax=ax)
    except Exception as e:
        print(e)
        raise RuntimeError("Failed to visualise tree structure.")


def visualise_ttn(
    tree: ttn_type,
    prog: str = "dot",
    ax=None,
    node_size: int = 300,
    linewidth: int = 3,
    bond_prop: Optional[str] = None,
    colourmap: str = "viridis",
    label_all_bonds: bool = False,
) -> None:
    """A function for plotting an ttn object

    :param tree: The tree to be visualised
    :type tree: ttn_type
    :param prog: The visualisation function used with graphviz_layout (default: "dot")
    :type prog: str, optional
    :param ax: Draw the tree in specified Matplotlib axes
    :type ax: Matplotlib Axes object, optional
    :param node_size: The node size used for visualisation (default: 500)
    :type node_size: int, optional
    :param linewidth: The width of edges connecting nodes (default: 3)
    :type linewidth: int, optional
    :param bond_prop: A string specifying the bond property to visualise for this system
    :type bond_prop: {"bond dimension", "bond capacity"}, optional
    :param colourmap: An optional name for colour map to use when plotting bond properties. (Default: 'viridis')
    :type colourmap: str, optional
    :param label_all_bonds: An optional boolean specifying whether or not to label all bonds in the TTN.  If False only the bonds with the maximum and minimum value of the property will be labelled. (Default: False)
    :type label_all_bonds: bool, optional
    """
    try:
        import networkx as nx
        from networkx.drawing.nx_pydot import graphviz_layout

        G = nx.Graph()

        # add all of the nodes to the tree
        for i, n in enumerate(tree):
            G.add_node(i)

        bonds = tree.bonds()
        for bond in bonds:
            G.add_edge(bond[0], bond[1])

        # get the bond properties that are to be visualised
        set_colours = False
        prop = None
        if isinstance(bond_prop, str):
            if bond_prop == "bond dimension":
                prop = tree.bond_dimensions()
                set_colours = True

            elif bond_prop == "bond capacity":
                prop = tree.bond_capacities()
                set_colours = True
            else:
                set_colours = False

        pos = graphviz_layout(G, prog=prog, root=0)
        if not set_colours:
            nx.draw(G, pos, node_size=node_size, width=linewidth, ax=ax)
        else:
            # now we get colours from the edge properties

            # first get the max and min values so we can normalise the colour range
            edges = G.edges()
            max_edge = max(prop, key=prop.get)
            max_val = prop[max_edge]
            min_edge = min(prop, key=prop.get)
            min_val = prop[min_edge]

            if label_all_bonds:
                edge_labels = prop
            else:
                edge_labels = {max_edge: str(max_val), min_edge: str(min_val)}

            if colourmap is not None:
                try:
                    import matplotlib as mpl

                    cmap = mpl.cm.get_cmap(colourmap)
                    colours = [cmap((prop[(v, u)]) / max_val) for u, v in edges]
                    nx.draw(
                        G,
                        pos,
                        node_size=node_size,
                        width=linewidth,
                        edge_color=colours,
                        ax=ax,
                    )
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                except Exception:
                    nx.draw(G, pos, node_size=node_size, width=linewidth, ax=ax)
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            else:
                nx.draw(G, pos, node_size=node_size, width=linewidth, ax=ax)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    except Exception as e:
        print(e)
        raise RuntimeError("Failed to visualise tree structure.")


def visualise_tree(
    tree: ntree | ttn_type | ms_ttn_type,
    prog: str = "dot",
    ax=None,
    node_size: int = 300,
    linewidth: int = 3,
    add_labels: bool = True,
    bond_prop: Optional[str] = None,
    colourmap: str = "viridis",
    label_all_bonds: bool = False,
) -> None:
    """A function for plotting a tree structured object

    :param tree: The tree to be visualised
    :type tree: ntree | ttn_type | ms_ttn_type
    :param prog: The visualisation function used with graphviz_layout (default: "dot")
    :type prog: str, optional
    :param ax: Draw the tree in specified Matplotlib axes
    :type ax: Matplotlib Axes object, optional
    :param node_size: The node size used for visualisation (default: 500)
    :type node_size: int, optional
    :param linewidth: The width of edges connecting nodes (default: 3)
    :type linewidth: int, optional
    :param add_labels: Whether or not to include the ntree node labels.  This option is only used if tree is an ntree. (default: True)
    :type add_labels: bool, optional
    :param bond_prop: A parameter specifying the bond property to plot for a ttn or ms_ttn.  See :meth:`visualise_ttn` or :meth:`visualise_ms_ttn` for more details.
    :param colourmap: An optional name for colour map to use when plotting bond properties. (Default: 'viridis')
    :type colourmap: str, optional
    :param label_all_bonds: An optional boolean specifying whether or not to label all bonds in the TTN.  If False only the bonds with the maximum and minimum value of the property will be labelled. (Default: False)
    :type label_all_bonds: bool, optional
    """

    from pyttn import is_ttn

    if isinstance(tree, ntree):
        visualise_ntree(
            tree,
            prog=prog,
            ax=ax,
            node_size=node_size,
            linewidth=linewidth,
            add_labels=add_labels,
        )
    elif isinstance(tree, ttn_type):
        visualise_ttn(
            tree,
            prog=prog,
            ax=ax,
            node_size=node_size,
            linewidth=linewidth,
            bond_prop=bond_prop,
            colourmap=colourmap,
            label_all_bonds=label_all_bonds,
        )
