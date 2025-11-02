from collections import defaultdict
from typing import List

import gradio as gr

from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk
from graphgen.models import LightRAGKGBuilder, OpenAIClient
from graphgen.utils import run_concurrent


async def build_kg(
    llm_client: OpenAIClient,
    kg_instance: BaseGraphStorage,
    chunks: List[Chunk],
    progress_bar: gr.Progress = None,
    namespace: str = None
):
    """
    :param llm_client: Synthesizer LLM model to extract entities and relationships
    :param kg_instance
    :param chunks
    :param progress_bar: Gradio progress bar to show the progress of the extraction
    :return:
    """

    kg_builder = LightRAGKGBuilder(llm_client=llm_client, max_loop=3)

    results = await run_concurrent(
        kg_builder.extract,
        chunks,
        desc="[2/4]Extracting entities and relationships from chunks",
        unit="chunk",
        progress_bar=progress_bar,
    )

    nodes = defaultdict(list)
    edges = defaultdict(list)
    for n, e in results:
        for k, v in n.items():
            nodes[k].extend(v)
        for k, v in e.items():
            edges[tuple(sorted(k))].extend(v) # Bidirectional encoding means that src can point to tgt, and tgt can also point to src.

    await run_concurrent(
        lambda kv: kg_builder.merge_nodes(kv, kg_instance=kg_instance),
        list(nodes.items()),
        desc="Inserting entities into storage",
    )

    await run_concurrent(
        lambda kv: kg_builder.merge_edges(kv, kg_instance=kg_instance),
        list(edges.items()),
        desc="Inserting relationships into storage",
    )

    graph = await kg_instance.get_graph()  # get NetworkX graph
    import networkx as nx
    # save to GraphML file
    nx.write_graphml(graph, f"/data1/nuist_llm/MF-Agent/mf_agent/data/graph/{namespace}.graphml")
    # # save to GEXF file（using Gephi vis）
    # nx.write_gexf(graph, "/data1/nuist_llm/mf-agent/GraphGen-main/build_data/data/knowledge_graph.gexf")

    return kg_instance
