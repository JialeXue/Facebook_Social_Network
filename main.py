import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from algorithms import *

if __name__ == '__main__':
    filename = "3. data.txt"
    # read the data into a dict form
    graph = defaultdict(list)
    with open(filename, 'r', encoding="UTF-8") as file:
        for line in file:
            node, neighbor = line.strip().split()
            graph[node].append(neighbor)
    # construct undirected graph
    for node, neighbors in list(graph.items()):
        for neighbor in neighbors:
            if node not in graph[neighbor]:
                graph[neighbor].append(node)
    # networkx graph
    g = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            g.add_edge(node, neighbor)

    betweenness = node_betweenness_cent(graph)
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

    pr_values = pagerank_cent(graph, 0.85, 0.15)
    sorted_pr = sorted(pr_values.items(), key=lambda x: x[1], reverse=True)[:10]

    with open('top_10_nodes.txt', 'w') as f:
        betweenness_nodes = [node for node, value in sorted_betweenness]
        f.write(' '.join(betweenness_nodes) + '\n')

        pr_nodes = [node for node, value in sorted_pr]
        f.write(' '.join(pr_nodes))

