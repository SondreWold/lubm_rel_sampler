import rdflib
import json
from tqdm import tqdm
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx
import numpy as np
from rdflib.term import Literal
from collections import defaultdict
import argparse

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-k', '--hops', type=int, default=4, help='The number of hops to sample for')
    args.add_argument('-s', '--samples', type=int, default=4, help='The number of samples to do per department for')
    return args.parse_args()

# Modification of the generate_random_paths from Networkx so that it handles rows with zero transition probabilities.
def generate_random_paths(
    G, sample_size, path_length=5, index_map=None, weight="weight"
):
    # Calculate transition probabilities between
    # every pair of vertices according to Eq. (3)
    adj_mat = nx.to_numpy_array(G, weight=weight)
    inv_row_sums = np.reciprocal(adj_mat.sum(axis=1)).reshape(-1, 1)
    transition_probabilities = adj_mat * inv_row_sums

    node_map = np.array(G)
    num_nodes = G.number_of_nodes()

    for path_index in range(sample_size):
        # Sample current vertex v = v_i uniformly at random
        node_index = np.random.randint(0, high=num_nodes)
        valid_node = False
        while valid_node == False:
            if np.sum(adj_mat[node_index]) == 0.0:
                node_index = np.random.randint(0, high=num_nodes)
            else:
                valid_node = True
            
        node = node_map[node_index]

        # Add v into p_r and add p_r into the path set
        # of v, i.e., P_v
        path = [node]

        # Build the inverted index (P_v) of vertices to paths
        if index_map is not None:
            if node in index_map:
                index_map[node].add(path_index)
            else:
                index_map[node] = {path_index}

        starting_index = node_index
        for _ in range(path_length):
            # Randomly sample a neighbor (v_j) according
            # to transition probabilities from ``node`` (v) to its neighbors
            neighbor_index = np.random.choice(
                num_nodes, p=transition_probabilities[starting_index]
            )

            # Set current vertex (v = v_j)
            starting_index = neighbor_index

            # Add v into p_r
            neighbor_node = node_map[neighbor_index]
            path.append(neighbor_node)

            # Add p_r into P_v
            if index_map is not None:
                if neighbor_node in index_map:
                    index_map[neighbor_node].add(path_index)
                else:
                    index_map[neighbor_node] = {path_index}

        yield path

if __name__ == '__main__':
    args = parse_args()
    relation_distribution = defaultdict(int)
    for i in range(15): # Number of departments
        g = rdflib.Graph()
        g.parse(f'./data/uni{i}.owl')

        tmp = rdflib_to_networkx_multidigraph(g)
        KG = nx.MultiDiGraph()

        for i, e in enumerate(tmp.edges):
            head, tail, pred = e
            if isinstance(head, Literal) or isinstance(tail, Literal):
                continue
            head = str(head)
            tail = str(tail)
            pred = str(pred)
            KG.add_edge(head, tail, type=pred, key=pred)

        steps = args.hops
        samples = args.samples
        for _ in tqdm(range(samples)):
            has_path = False
            while has_path == False:
                try:
                    random_path = [x for x in generate_random_paths(KG, 1, steps)][0]
                    has_path = True
                except:
                    continue

            edges = []
            for i in range(0, steps):
                edge = KG.get_edge_data(random_path[i], random_path[i+1])
                edges.append(list(edge.keys())[0])

            relation_distribution[" ".join(edges)] += 1

    with open(f'distribution_{steps}-hops_{15*samples}samples.json', 'w') as f:
        json.dump(relation_distribution, f)
