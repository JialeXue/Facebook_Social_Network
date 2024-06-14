import numpy as np

def node_betweenness_cent(adj_list):
    # initialise
    betweenness = {key: 0 for key in adj_list}
    for s in adj_list:
        stack = []
        # initialise the predecessor node list
        P = {key: [] for key in adj_list}
        # num of shortest path
        sigma = dict.fromkeys(adj_list, 0)
        # distance
        D = dict.fromkeys(adj_list, -1)
        sigma[s] = 1
        D[s] = 0
        # BFS queue
        Q = [s]

        # BFS
        while Q:
            v = Q.pop(0)
            stack.append(v)
            for w in adj_list[v]:
                if D[w] < 0:
                    Q.append(w)
                    D[w] = D[v] + 1
                if D[w] == D[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        # accumulation
        # initialise dependency scores
        delta = dict.fromkeys(adj_list, 0)
        while stack:
            w = stack.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Undirected graph/2
    for v in betweenness:
        betweenness[v] /= 2

    return betweenness


def pagerank_cent(graph, alpha, beta):
    nodes = sorted(graph.keys())
    n = len(nodes)
    A = np.zeros((n, n))

    # adjacency matrix
    for node, neighbors in graph.items():
        i = nodes.index(node)
        for neighbor in neighbors:
            if neighbor in nodes:
                j = nodes.index(neighbor)
                A[i][j] = 1

    # degree matrix
    D = np.diag(A.sum(axis=1))

    eps = 1e-6
    is_converged = False
    pr = np.ones(n) / n
    while not is_converged:
        previous_pr = pr.copy()
        pr = alpha * np.dot(A.T, np.linalg.inv(D).dot(pr)) + beta * np.ones(n)
        pr = pr / np.sum(pr)
        is_converged = np.sum(np.abs(previous_pr - pr)) < eps
    return {nodes[i]: pr_val for i, pr_val in enumerate(pr)}