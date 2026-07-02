import networkx as nx
import matplotlib.pyplot as plt
import xgi

def visualise_interaction_graph(
    graph,
    with_labels: bool = True,
    node_size: int = 500,
    font_size: int = 10,
    edge_width_scale: float = 1.0,
):
    """
    Visualise an InteractionGraph using NetworkX.

    :param graph: The interaction graph
    :type graph: InteractionGraph
    :param with_labels: Whether to draw node labels
    :type with_labels: bool
    :param node_size: Size of nodes in the plot
    :type node_size: int
    :param font_size: Font size for labels
    :type font_size: int
    :param edge_width_scale: Scaling factor for edge thickness
    :type edge_width_scale: float
    """

    G_nx = nx.Graph()

    # add nodes
    for node in graph.nodes:
        G_nx.add_node(node)

    # add edges with weights
    for (u, v), data in graph.edges.items():
        u_, v_ = [u, v]
        G_nx.add_edge(u_, v_, weight=data["weight"])

    pos = nx.spring_layout(G_nx)

    # edge widths proportional to weight
    weights = [d["weight"] for _, _, d in G_nx.edges(data=True)]
    widths = [edge_width_scale * w for w in weights]

    nx.draw(
        G_nx,
        pos,
        with_labels=with_labels,
        node_size=node_size,
        width=widths,
        font_size=font_size,
    )

    plt.title("Interaction Graph")


def visualise_interaction_hypergraph(
    hypergraph,
    with_labels: bool = True,
    node_size: int = 300,
):
    """
    Visualise an InteractionHypergraph using XGI.

    :param hypergraph: The interaction hypergraph
    :type hypergraph: InteractionHypergraph
    :param with_labels: Whether to draw node labels
    :type with_labels: bool
    :param node_size: Size of nodes in the plot
    :type node_size: int
    """

    H_xgi = xgi.Hypergraph()

    # add nodes
    for node in hypergraph.nodes:
        H_xgi.add_node(node)
        print(node)

    # add hyperedges
    H_xgi.add_edges_from([list(nodes) for nodes in hypergraph.hyperedges.items()])
    print([list(nodes) for nodes in hypergraph.hyperedges.items()])

    pos = xgi.barycenter_spring_layout(H_xgi, seed=1)
    xgi.draw(H_xgi, pos, hull=True, edge_lw=2, edge_ec='k')

    plt.title("Interaction Hypergraph")
