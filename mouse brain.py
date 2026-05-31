
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw
import random

# =========================
# 1. خواندن فایل شبکه
# =========================

G = nx.read_edgelist(
    "bn-mouse-kasthuri_graph_v4/bn-mouse-kasthuri_graph_v4.edges",
    nodetype=int
)

# =========================
# 2. اطلاعات اولیه شبکه
# =========================

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# =========================
# 3. ماتریس مجاورت
# =========================

A = nx.to_numpy_array(G)

plt.figure(figsize=(10, 10))
plt.imshow(A, cmap="binary")

plt.title("Adjacency Matrix - Mouse Brain Network")
plt.xlabel("Nodes")
plt.ylabel("Nodes")

plt.show()

# =========================
# 4. درجه هر گره
# =========================

degrees = dict(G.degree())

print("\nDegree of each node:\n")

for node, degree in degrees.items():
    print(f"Node {node}: Degree = {degree}")

# =========================
# 5. توزیع درجه
# =========================

degree_values = [d for n, d in G.degree()]

plt.figure(figsize=(8, 5))
plt.hist(degree_values, bins=30)

plt.xlabel("Degree")
plt.ylabel("Number of Nodes")

plt.title("Degree Distribution")

plt.show()

# =========================
# 6. خلاصه شبکه
# =========================

avg_degree = sum(degree_values) / len(degree_values)

print("\n========== NETWORK SUMMARY ==========")
print("Nodes =", G.number_of_nodes())
print("Edges =", G.number_of_edges())
print("Average Degree =", round(avg_degree, 3))

# =========================
# 7. بررسی Scale-Free
# =========================

fit = powerlaw.Fit(degree_values, verbose=False)

fig = fit.plot_pdf(color='b', linewidth=2)

fit.power_law.plot_pdf(
    color='r',
    linestyle='--',
    ax=fig
)

plt.title("Power Law Fit")
plt.xlabel("Degree")
plt.ylabel("P(k)")

plt.show()

gamma = fit.power_law.alpha

print("\n========== SCALE-FREE ANALYSIS ==========")
print(f"Gamma (α) = {gamma:.3f}")

R, p = fit.distribution_compare(
    'power_law',
    'exponential'
)

print(f"R = {R:.4f}")
print(f"p = {p:.4f}")

if R > 0 and p < 0.05:
    print("\nConclusion:")
    print("The power-law model fits significantly better than the exponential model.")
    print("The network shows strong evidence of scale-free behavior.")

elif R > 0 and p >= 0.05:
    print("\nConclusion:")
    print("The power-law model is preferred, but the evidence is not statistically strong.")
    print("The network may exhibit scale-free characteristics.")

else:
    print("\nConclusion:")
    print("The power-law model is not preferred over the exponential model.")
    print("There is insufficient evidence to classify the network as scale-free.")

# =========================
# 8. Clustering Coefficient
# =========================

avg_clustering = nx.average_clustering(G)

print("\n========== CLUSTERING ==========")
print("Average Clustering Coefficient =", avg_clustering)

# =========================
# 9. Average Path Length
# =========================

if nx.is_connected(G):

    avg_path = nx.average_shortest_path_length(G)

    print("\n========== PATH LENGTH ==========")
    print("Average Path Length =", avg_path)

else:

    largest_cc = max(
        nx.connected_components(G),
        key=len
    )

    G_largest = G.subgraph(largest_cc)

    avg_path = nx.average_shortest_path_length(G_largest)

    print("\n========== PATH LENGTH ==========")
    print("Graph is disconnected")
    print("Average Path Length (Largest Component) =", avg_path)

# =========================
# 10. Hub Analysis
# =========================

degree_centrality = nx.degree_centrality(G)

top_hubs = sorted(
    degree_centrality.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\n========== TOP 10 HUBS ==========")

for node, score in top_hubs:
    print(f"Node {node} : Centrality = {score:.4f}")

hub = max(
    degree_centrality,
    key=degree_centrality.get
)

print("\nMain Hub =", hub)
print("Centrality =", degree_centrality[hub])

# =========================
# 11. Fault Tolerance
# =========================

G_attack = G.copy()

largest_components = []

nodes = list(G_attack.nodes())
random.shuffle(nodes)

for i in range(100):

    G_attack.remove_node(nodes[i])

    largest_cc = max(
        nx.connected_components(G_attack),
        key=len
    )

    largest_components.append(len(largest_cc))

plt.figure(figsize=(8, 5))

plt.plot(largest_components)

plt.xlabel("Removed Nodes")
plt.ylabel("Largest Connected Component")

plt.title("Network Robustness")

plt.show()