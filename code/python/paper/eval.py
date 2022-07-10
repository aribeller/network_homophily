import networkx as nx

def compute_inequity(g, k):
    """Compute the proportion of minorities in the top k ranks of g"""
    node_pageranks = nx.pagerank(g)
    node_pageranks_sorted = sorted(node_pageranks.items(), key=lambda x: x[1], reverse=True)
    top_k = node_pageranks_sorted[:k]
    num_top_k_minority = sum([g.nodes[node_id]['m'] for (node_id, _) in top_k])

    return num_top_k_minority / k