import asyncio
from pyvis.network import Network
from graphgen.models import NetworkXStorage

async def main():
    kg_instance = NetworkXStorage(
        "/data1/nuist_llm/mf-agent/GraphGen-main/build_data/data",
        namespace="knowledge_graph" # save to knowledge_graph.graphml file
    )
    """
    # graph = await kg_instance.get_graph()  # get NetworkX graph
    # import networkx as nx
    # # save to GraphML file
    # nx.write_graphml(graph, "/data1/nuist_llm/mf-agent/GraphGen-main/build_data/data/knowledge_graph.graphml")
    # # save to GEXF file（using Gephi vis）
    # nx.write_gexf(graph, "/data1/nuist_llm/mf-agent/GraphGen-main/build_data/data/knowledge_graph.gexf")
    """
    graph = await kg_instance.get_graph()
    print(f"Loading Graph Success: \n Node Number: {graph.number_of_nodes()}\n Edge Number: {graph.number_of_edges()}")
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    for node, data in graph.nodes(data=True):
        net.add_node(node, label=node, title=str(data))

    for src, tgt, data in graph.edges(data=True):
        net.add_edge(src, tgt, title=str(data))
    net.show("/data1/nuist_llm/mf-agent/GraphGen-main/build_data/data/knowledge_graph.html", notebook=False)
    print("Vis of Graph have been generate to :knowledge_graph.html")

asyncio.run(main())