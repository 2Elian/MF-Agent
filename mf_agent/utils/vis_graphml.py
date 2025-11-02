import asyncio
from pyvis.network import Network
from graphgen.models import NetworkXStorage

async def main():
    kg_instance = NetworkXStorage(
        "/data1/nuist_llm/mf-agent/GraphGen-main/build_data/data",
        namespace="graphml文件的名称"
    )
    graph = await kg_instance.get_graph()
    print(f"✅ 加载图成功！节点数: {graph.number_of_nodes()}，边数: {graph.number_of_edges()}")
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    for node, data in graph.nodes(data=True):
        net.add_node(node, label=node, title=str(data))

    for src, tgt, data in graph.edges(data=True):
        net.add_edge(src, tgt, title=str(data))
    net.show("/data1/nuist_llm/mf-agent/GraphGen-main/build_data/data/knowledge_graph.html", notebook=False)
    print("✅ 可视化图已生成：knowledge_graph.html")

asyncio.run(main())