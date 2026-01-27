import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load the data
nodes = pd.read_csv("nodes.csv")
edges = pd.read_csv("edges.csv")
traffic = pd.read_csv("traffic.csv")

print(f"Loaded {len(nodes)} nodes, {len(edges)} edges, and {len(traffic)} traffic entries")

# Graph and Transition Matrix
# 1) Graph -> hyperlink between wiki articles
graph = nx.DiGraph()
graph.add_nodes_from(nodes["node"])
graph.add_edges_from(edges[["source", "target"]].values)

N = len(nodes)
node_index = nodes["node"].to_dict()
node_index = {node_index[k]: k for k in node_index}

# 2) Transition Matrix -> probability moving from j to i
M = np.zeros((N, N))
for u, v in graph.edges():
    if graph.out_degree(u) > 0:
        M[node_index[v], node_index[u]] = 1.0 / graph.out_degree(u)

print(f"Transition matrix M built with shape: {M.shape}")

# Damping Factor -> prob follow link VS random page
damping_factor = 0.85

# PageRank using network
pagerank = nx.pagerank(graph, alpha=damping_factor)
pagerank_vec = np.array([pagerank[n] for n in nodes["node"]])
nx.set_node_attributes(graph, pagerank, "pagerank")

# Power iteration:
G_matrix = damping_factor * M + (1 - damping_factor) / N * np.ones((N, N))
p = np.ones(N) / N # Page rank -> Everyone same possibilities
tol = 1e-10
max_iter = 1000

for i in range(max_iter):
    p_next = G_matrix @ p
    p_next /= np.linalg.norm(p_next)
    if np.linalg.norm(p_next - p, 2) < tol:
        print(f"Converged after {i} iterations")
        break
    p = p_next

pagerank_powit = p / np.sum(p) # Normalizzazione

# Compare results
corr = np.corrcoef(pagerank_vec, pagerank_powit)[0, 1]
l1_diff = np.linalg.norm(pagerank_vec - pagerank_powit, 1)

print("\n--- PageRank Comparison ---")
print(f"Pearson correlation: {corr:.6f}")
print(f"L1 difference:       {l1_diff:.6e}")

plt.plot(pagerank_vec, 'o-')
plt.plot(pagerank_powit, 'x-')

plt.figure(figsize=(12, 10))

pos = nx.spring_layout(graph, k=0.5, seed=42)

# Draw nodes
nx.draw_networkx_nodes(
    graph,
    pos,
    node_size=pagerank_vec * 5000,
    node_color="skyblue",
    alpha=0.8,
    edgecolors="gray",
)

# Draw edges
nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.4)

# Draw labels (of most important pages)optional for smaller graphs
nx.draw_networkx_labels(
    graph,
    pos,
    labels={
        n: n for n in nodes["node"] if pagerank[n] > np.quantile(pagerank_vec, 0.9)
    },
    font_color="k",
    font_size=10,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
)

plt.title("Wikipedia Machine Learning Graph\nNode size = PageRank", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()

traffic["pagerank"] = traffic["node"].map(pagerank)

plt.figure(figsize=(7, 6))
plt.scatter(traffic.traffic, traffic.pagerank, alpha=0.7)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Wikipedia Traffic (log scale)")
plt.ylabel("PageRank (log scale)")
plt.title("Correlation between PageRank and Traffic")
plt.tight_layout()
plt.show()

corr_traffic = np.corrcoef(traffic.traffic, traffic.pagerank)[0, 1]
print(f"Correlation between traffic and pagerank: {corr_traffic:.3f}")

print("\nTop 10 pages by PageRank:")
print(traffic.nlargest(10, "pagerank")[["node", "pagerank", "traffic"]])