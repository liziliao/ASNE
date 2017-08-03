import numpy as np
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score


def calculate_distance( embeddings, type): # N * emb_size
    if type == 'euclidean_distances':
        Y_predict = -1.0 * euclidean_distances(embeddings, embeddings)
        return Y_predict
    if type == 'cosine_similarity':
        Y_predict = cosine_similarity(embeddings, embeddings)
        return Y_predict

def norm(a):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * a[i]
    return math.sqrt(sum)

def cosine_similarity( a,  b):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * b[i]
    return sum/(norm(a) * norm(b))

def evaluate_ROC(X_test, Embeddings):
    y_true = [ X_test[i][2] for i in range(len(X_test))]
    y_predict = [ cosine_similarity(Embeddings[X_test[i][0],:], Embeddings[X_test[i][1], :]) for i in range(len(X_test))]
    roc = roc_auc_score(y_true, y_predict)
    if roc < 0.5:
        roc = 1 - roc
    return roc


def evaluate_MAP( node_neighbors_map, Embeddings, distance_measure):
    '''
    given the embeddings of nodes and the node_neighbors, return the MAP value
    :param node_neighbors_map: [node_id : neighbors_ids]
    :param nodes: a dictionary, ['node_id']--len(nodes) of id for nodes, one by one; ['node_attr']--a list of attrs for corresponding nodes
    :param Embeddings:  # nodes_number * (id_dim + attr_dim), row sequence is the same as nodes['node_id']
    :return: MAP value
    '''
    MAP = .0
    Y_true = np.zeros((len(node_neighbors_map), len(node_neighbors_map)))
    for node in node_neighbors_map:
        # prepare the y_true
        for neighbor in node_neighbors_map[node]:
            Y_true[node][neighbor] = 1

    print distance_measure
    Y_predict = calculate_distance(Embeddings,distance_measure)
    for node in node_neighbors_map:
        MAP +=  average_precision_score(Y_true[node,:], Y_predict[node,:])

    return MAP/len(node_neighbors_map)
