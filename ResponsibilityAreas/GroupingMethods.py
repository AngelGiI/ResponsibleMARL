################################################################################################
###   Main script where the grouping methods are applied to get the Responsibility areas.    ###
#                                                                                              #
#     These add 3 new input params:                                                            #
#       - Number of Clusters: 1 to N, where N is the  total number of substations.             #
#       - Clustering Method: to choose from KMeans, Spectral, Louvain and Hierarchical.        #
#       - Adjacency Matrix: can be Unweighted, Action, Congestion, or ActionCongestion         #
#                                                                                              #
################################################################################################

import abc
import networkx as nx
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sknetwork.clustering import Louvain
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
from collections import defaultdict
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Hardcoded for now. Results from running "AvgLineCongestion.py"
# Average congestion per line for the 14-bus case on 37800 timesteps with DoNothingAgent
average_congestion_case14 = [0.31120053, 0.3033225, 0.30382577, 0.23219728, 0.67197347, 0.19390275,
    0.40178862, 0.64066064, 0.5332323,  0.7970488, 0.14247213, 0.31393358, 0.36672422, 
    0.47334263, 0.4611472,  0.43447793, 0.41216594, 0.5247531, 0.46909025, 0.4043246]

class BaseClustering(metaclass=abc.ABCMeta):
    def __init__(self, obs, num_clusters, actions, mask=0, adjacency_type='unweighted'):
        self.graph = self.create_graph(obs)
        self.num_clusters = num_clusters
        self.num_substations = len(obs.grid_layout)
        self.distance_matrix = self.compute_distance_matrix()
        self.actions = np.array(actions)
        self.mask = mask
        self.adjacency_type = adjacency_type
        self.congestion = average_congestion_case14

    def create_graph(self, obs):
        graph = nx.Graph()
        for line_or, line_ex in zip(obs.line_or_to_subid, obs.line_ex_to_subid):
            graph.add_edge(line_or, line_ex)
        return graph
    
    def compute_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.num_substations, self.num_substations))
        for i, j in self.graph.edges():
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
        return adjacency_matrix
    
    def normalize_matrix(self, matrix):
        max_val = np.max(matrix)
        if max_val > 0:
            return matrix / max_val
        return matrix
    
    def compute_congestion_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.num_substations, self.num_substations))
        line_id = 0
        for i, j in self.graph.edges():
            adjacency_matrix[i][j] = self.congestion[line_id]
            adjacency_matrix[j][i] = self.congestion[line_id]
            line_id += 1
        return self.normalize_matrix(adjacency_matrix)

    def compute_actions_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.num_substations, self.num_substations))
        masked_actions = np.where(self.actions <= self.mask, 0, self.actions)
        zero_action_substations = np.where(masked_actions == 0)[0]

        line_id = 0
        for i, j in self.graph.edges():
            if masked_actions[i] == 0 and masked_actions[j] == 0:
                adjacency_matrix[i][j] = adjacency_matrix[j][i] = 0.1  # Small weight for masked-to-masked connections
            else:
                if masked_actions[i] == 0 or masked_actions[j] == 0:
                    adjacency_matrix[i][j] = adjacency_matrix[j][i] = max(self.actions[i], self.actions[j])
                else:
                    adjacency_matrix[i][j] = adjacency_matrix[j][i] = (self.actions[i] + self.actions[j]) / 2
            line_id += 1
        return self.normalize_matrix(adjacency_matrix)
    
    def compute_actions_congestion_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.num_substations, self.num_substations))
        masked_actions = np.where(self.actions <= self.mask, 0, self.actions)
        zero_action_substations = np.where(masked_actions == 0)[0]

        line_id = 0
        for i, j in self.graph.edges():
            if masked_actions[i] == 0 and masked_actions[j] == 0:
                adjacency_matrix[i][j] = adjacency_matrix[j][i] = self.congestion[line_id]
            else:
                if masked_actions[i] == 0 or masked_actions[j] == 0:
                    adjacency_matrix[i][j] = adjacency_matrix[j][i] = self.congestion[line_id] * max(self.actions[i], self.actions[j])
                else:
                    adjacency_matrix[i][j] = adjacency_matrix[j][i] = self.congestion[line_id] * (self.actions[i] + self.actions[j]) / 2
            line_id += 1
        return self.normalize_matrix(adjacency_matrix)   

    def select_adjacency_matrix(self):
        if self.adjacency_type == 'congestion':
            return self.compute_congestion_adjacency_matrix()
        elif self.adjacency_type == 'actions':
            return self.compute_actions_adjacency_matrix()
        elif self.adjacency_type == 'actions_congestion':
            return self.compute_actions_congestion_adjacency_matrix()
        else:
            return self.compute_adjacency_matrix()

    def compute_distance_matrix(self):
        distance_matrix = np.full((self.num_substations, self.num_substations), float('inf'))
        for sub in range(self.num_substations):
            lengths = nx.single_source_shortest_path_length(self.graph, sub)
            for target, length in lengths.items():
                distance_matrix[sub][target] = length
        return distance_matrix
    
    @abc.abstractmethod
    def perform_clustering(self):
        pass

class KMeansClustering(BaseClustering):
    def perform_clustering(self):
        adjacency_matrix = self.select_adjacency_matrix()
        clustering_model = KMeans(n_clusters=self.num_clusters)
        clusters = clustering_model.fit_predict(adjacency_matrix)
        cluster_dict = defaultdict(list)
        for substation, cluster_id in enumerate(clusters):
            cluster_dict[cluster_id].append(substation)
        return list(cluster_dict.values())

class Spectral(BaseClustering):
    def perform_clustering(self):
        adjacency_matrix = self.select_adjacency_matrix()
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        with np.errstate(divide='ignore', invalid='ignore'):
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
            D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
            D_inv_sqrt[np.isnan(D_inv_sqrt)] = 0
        
        L_normalized = np.identity(self.num_substations) - D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt

        eigvals, eigvecs = np.linalg.eigh(L_normalized)
        idx = np.argsort(eigvals)[:self.num_clusters]
        spectral_embedding = eigvecs[:, idx]
        
        spectral_embedding = normalize(spectral_embedding, norm='l2')
        
        clustering_model = KMeans(n_clusters=self.num_clusters)
        clusters = clustering_model.fit_predict(spectral_embedding)
        
        cluster_dict = defaultdict(list)
        for substation, cluster_id in enumerate(clusters):
            cluster_dict[cluster_id].append(substation)
        return list(cluster_dict.values())
    
    def compute_eigenvalues(self):
        adjacency_matrix = self.select_adjacency_matrix()
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        with np.errstate(divide='ignore', invalid='ignore'):
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
            D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
            D_inv_sqrt[np.isnan(D_inv_sqrt)] = 0
        
        L_normalized = np.identity(self.num_substations) - D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt

        eigvals, eigvecs = np.linalg.eigh(L_normalized)
        return eigvals[:8], eigvecs[:, :8]
    
    def plot_eigenvalues_and_eigengaps(eigvals):
        plt.style.use('ggplot')
        plt.rcParams.update({
            'font.size': 12,               # Font size
            'axes.labelsize': 22,          # Axis label size
            'axes.titlesize': 24,          # Title size
            'xtick.labelsize': 12,         # X tick label size
            'ytick.labelsize': 12,         # Y tick label size
            'figure.figsize': (8, 6),      # Figure size
            'lines.linewidth': 2,          # Line width
            'grid.alpha': 0.8,             # Grid transparency
            'grid.linestyle': '--',        # Grid line style
        })
        plt.figure(figsize=(10, 6))
        plt.plot(eigvals, marker='o', linestyle='-')
        plt.title('Eigenvalues of the Normalized Laplacian')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.grid(True)
        plt.show()
        
        eigengaps = np.diff(eigvals)
        plt.figure(figsize=(10, 6))
        plt.plot(eigengaps, marker='o', linestyle='-')
        plt.title('Eigengaps of the Normalized Laplacian')
        plt.xlabel('Index')
        plt.ylabel('Eigengap')
        plt.grid(True)
        plt.show()

class LouvainClustering(BaseClustering):
    def perform_clustering(self):
        adjacency_matrix = self.select_adjacency_matrix()
        louvain = Louvain()
        labels = louvain.fit_predict(csr_matrix(adjacency_matrix))

        cluster_dict = defaultdict(list)
        for substation, cluster_id in enumerate(labels):
            cluster_dict[cluster_id].append(substation)
        return list(cluster_dict.values())

class HierarchicalClustering(BaseClustering):
    def compute_weighted_distance_matrix(self):
        adjacency_matrix = self.select_adjacency_matrix()
        
        # Initialize the distance matrix with the adjacency matrix values
        distance_matrix = np.full((self.num_substations, self.num_substations), float('inf'))
        for i in range(self.num_substations):
            for j in range(self.num_substations):
                if i == j:
                    distance_matrix[i][j] = 0
                elif adjacency_matrix[i][j] > 0:
                    distance_matrix[i][j] = adjacency_matrix[i][j]

        # Floyd-Warshall algorithm to compute shortest paths between all pairs of nodes
        for k in range(self.num_substations):
            for i in range(self.num_substations):
                for j in range(self.num_substations):
                    if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                        distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]

        return distance_matrix

    def perform_clustering(self, linkage_method='ward'):
        # Compute the weighted distance matrix
        adjacency_matrix = self.select_adjacency_matrix()
        distance_matrix = self.compute_weighted_distance_matrix()

        # Convert the full distance matrix to a condensed distance matrix
        condensed_distance_matrix = squareform(distance_matrix)

        # Perform hierarchical/agglomerative clustering
        Z = linkage(condensed_distance_matrix, method=linkage_method)

        if self.num_clusters:
            # Cut the dendrogram at the desired number of clusters to form n_clusters
            cluster_labels = fcluster(Z, self.num_clusters, criterion='maxclust')
            cluster_dict = defaultdict(list)
            for substation, cluster_id in enumerate(cluster_labels):
                cluster_dict[cluster_id].append(substation)
            return list(cluster_dict.values())
        else:
            return Z

    def plot_dendrogram(self, Z, **kwargs):
        plt.rcParams.update({
            'font.size': 20,               # Font size
            'axes.labelsize': 24,          # Axis label size
            'axes.titlesize': 24,          # Title size
            'xtick.labelsize': 26,         # X tick label size
            'ytick.labelsize': 18,         # Y tick label size
            'figure.figsize': (8, 6),      # Figure size
            'lines.linewidth': 2,          # Line width
            'grid.alpha': 0.8,             # Grid transparency
            'grid.linestyle': '--',        # Grid line style
        })
        plt.figure()
        dendrogram(Z, **kwargs)
        plt.xlabel('Substations')
        plt.ylabel('Inter-cluster Distance')
        plt.grid(True)
        # Explicitly set the font size for the xtick labels
        plt.setp(plt.gca().get_xticklabels(), fontsize=18)
        plt.show()

# class KernighanLinClustering(BaseClustering):
#     def perform_clustering(self):
#         nodes = list(self.graph.nodes)
#         partitions = [nodes]

#         while len(partitions) < self.num_clusters:
#             new_partitions = []
#             for partition in partitions:
#                 if len(partition) > 1:
#                     subgraph = self.graph.subgraph(partition)
#                     part1, part2 = nx.algorithms.community.kernighan_lin_bisection(subgraph)
#                     new_partitions.extend([list(part1), list(part2)])
#                 else:
#                     new_partitions.append(partition)
#             partitions = new_partitions

#         cluster_dict = defaultdict(list)
#         for idx, partition in enumerate(partitions):
#             for node in partition:
#                 cluster_dict[idx].append(node)

#         return list(cluster_dict.values())

# class RecursiveBisectionClustering(BaseClustering):
#     def balance_bisection(self, subgraph, max_iterations=10):
#         best_cut = None
#         best_balance = float('inf')
#         for _ in range(max_iterations):
#             part1, part2 = nx.algorithms.community.kernighan_lin_bisection(subgraph)
#             balance = abs(len(part1) - len(part2))
#             if balance < best_balance:
#                 best_balance = balance
#                 best_cut = (part1, part2)
#             if best_balance == 0:
#                 break
#         return best_cut

#     def perform_clustering(self):
#         nodes = list(self.graph.nodes)
#         partitions = [nodes]

#         while len(partitions) < self.num_clusters:
#             new_partitions = []
#             for partition in partitions:
#                 if len(partition) > 1:
#                     subgraph = self.graph.subgraph(partition)
#                     part1, part2 = self.balance_bisection(subgraph)
#                     new_partitions.extend([list(part1), list(part2)])
#                 else:
#                     new_partitions.append(partition)
#             partitions = new_partitions

#         if len(partitions) > self.num_clusters:
#             while len(partitions) > self.num_clusters:
#                 partitions.sort(key=len)
#                 part1 = partitions.pop(0)
#                 part2 = partitions.pop(0)
#                 merged_partition = list(part1) + list(part2)
#                 partitions.append(merged_partition)

#         cluster_dict = defaultdict(list)
#         for idx, partition in enumerate(partitions):
#             for node in partition:
#                 cluster_dict[idx].append(node)

#         return list(cluster_dict.values())

class ClusteringManager:
    def __init__(self, method, obs, num_clusters, actions, mask=0, adjacency_type='unweighted'):
        self.method = method.lower()
        self.obs = obs
        self.num_clusters = num_clusters
        self.actions = actions
        self.mask = mask
        self.adjacency_type = adjacency_type

    def get_clustering(self):
        if self.method == 'kmeans':
            return KMeansClustering(self.obs, self.num_clusters, self.actions, self.mask, self.adjacency_type)
        elif self.method == 'spectral':
            return Spectral(self.obs, self.num_clusters, self.actions, self.mask, self.adjacency_type)
        elif self.method == 'louvain':
            return LouvainClustering(self.obs, self.num_clusters, self.actions, self.mask, self.adjacency_type)
        elif self.method == 'hierarchical':
            return HierarchicalClustering(self.obs, self.num_clusters, self.actions, self.mask, self.adjacency_type)
        # elif self.method == 'recursive_bisection':
        #     return RecursiveBisectionClustering(self.obs, self.num_clusters, self.actions, self.mask, self.adjacency_type)
        # elif self.method == 'kernighan_lin':
        #     return KernighanLinClustering(self.obs, self.num_clusters, self.actions, self.mask, self.adjacency_type)
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

    def perform_clustering(self):
        clustering = self.get_clustering()
        return clustering.perform_clustering()

    def plot_dendrogram(self, Z, **kwargs):
        if self.method == 'hierarchical':
            clustering = self.get_clustering()
            clustering.plot_dendrogram(Z, **kwargs)