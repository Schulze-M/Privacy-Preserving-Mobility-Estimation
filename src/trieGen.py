import json
import networkx as nx
import matplotlib.pyplot as plt

# Load the JSON
with open('trie_noDP.json') as f:
    trie = json.load(f)

# Create directed graph
G = nx.DiGraph()

# Recursive function to add nodes and edges
def add_nodes_edges(node, prefix):
    count = node.get('count', 0)
    G.add_node(prefix, label=f"{prefix}\n{count:.2f}")
    for child_char, child_node in node.get('children', {}).items():
        child_prefix = prefix + child_char
        G.add_node(child_prefix, label=f"{child_char}\n{child_node.get('count', 0):.2f}")
        G.add_edge(prefix, child_prefix)
        add_nodes_edges(child_node, child_prefix)

# Add top-level nodes and edges
for char, node in trie.items():
    node_prefix = char
    count = node.get('count', 0)
    G.add_node(node_prefix, label=f"{char}\n{count:.2f}")
    add_nodes_edges(node, char)

# Attempt a hierarchical layout using Graphviz; fallback to spring layout
try:
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
except Exception:
    pos = nx.spring_layout(G, k=0.5, iterations=50)

# Draw the trie
plt.figure(figsize=(20, 20))
labels = nx.get_node_attributes(G, 'label')
nx.draw(G, pos, labels=labels, with_labels=True, node_size=200, font_size=8, arrowsize=10)
plt.title("Trie Visualization")
plt.axis('off')
plt.tight_layout()
plt.savefig('trie_visualization.png', dpi=300)
