import numpy as np
import os
import random
from random import shuffle

class LoadData( object ):
    '''given the path of data, return the data format for ESNA
    :param path
    return:
     X: a dictionary, ['data_id_list']--len(links) of id for nodes in links ; ['data_attr_list']--a list of attrs for corresponding nodes;
                     ['data_label_list']--len(links) of neighbor for corresponding nodes

     nodes: a dictionary, ['node_id']--len(nodes) of id for nodes, one by one; ['node_attr']--a list of attrs for corresponding nodes
    '''

    # Three files are needed in the path
    def __init__(self, path, random_seed):
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.path = path
        self.linkfile = path + "doublelink.edgelist"
        self.testlinkfile = path + "test_pairs.txt"
        self.attrfile = path + "attr_info.txt"
        #self.vocabfile = path + "vocab.txt"
        self.node_map = {} # [node_name: id] for map node to id inside the program, based on links since some nodes might not have attributes
        self.nodes = {}
        self.X = {}
        self.X_test = []# a list of 3-element lists, read from test_link.txt
        self.node_neighbors_map = {} # [nodeid: neighbors_set] each node id maps to its neighbors set
        self.construct_nodes()
        self.construct_X()
        self.construct_node_neighbors_map()
        self.read_test_link()

    def readvocab(self):
        f = open(self.attrfile)
        self.vocab = {}
        line = f.readline()
        i = 0
        while line:
            items = line.strip().split(' ')
            for item in items[1:]:
                if item not in self.vocab:
                    self.vocab[ item ] = i
                    i = i + 1
            line = f.readline()
        f.close()
        self.attr_M = i
        print("attr_M:", self.attr_M)

    def construct_nodes(self):
        '''construct the dictionary '''
        self.readvocab()
        f = open(self.attrfile)
        i = 0
        self.nodes['node_id'] = []
        self.nodes['node_attr'] = []
        line = f.readline()
        while line:
            line = line.strip().split(' ')
            self.node_map[int(line[0])] = i # map the node
            self.nodes['node_id'].append(i) # only put id in nodes, not the original name
            vs = np.zeros(self.attr_M)
            for attr in line[1:]:
                if len(attr) > 0:
                    vs[self.vocab[attr]] = 1
            self.nodes['node_attr'].append(vs)
            i = i + 1
            line = f.readline()
        f.close()
        self.id_N = i
        print("id_N:", self.id_N)

    def read_link(self): # read link file to a list of links
        f = open(self.linkfile)
        self.links = []
        line = f.readline()
        while line:
            line = line.strip().split(' ')
            link = [int(line[0]), int(line[1])]
            self.links.append(link)
            line = f.readline()
        f.close()

    def construct_X(self):
        self.read_link()
        self.X['data_id_list'] = np.ndarray(shape=(len(self.links)), dtype=np.int32)
        self.X['data_attr_list'] = np.ndarray(shape=(len(self.links), self.attr_M), dtype=np.float32)
        self.X['data_label_list'] = np.ndarray(shape=(len(self.links), 1), dtype=np.int32)

        for i in range(len(self.links)):
            self.X['data_id_list'][i] = self.node_map[self.links[i][0]]
            self.X['data_attr_list'][i] =  self.nodes['node_attr'][ self.node_map[self.links[i][0]] ]  # dimension need to change to  self.attr_dim
            self.X['data_label_list'][i, 0] = self.node_map[self.links[i][1]]  # one neighbor of the node

    def construct_node_neighbors_map(self):
        for link in self.links:
            if self.node_map[ link[0] ] not in self.node_neighbors_map:
                self.node_neighbors_map[self.node_map[ link[0] ]] = set( [self.node_map[ link[1] ]] )
            else:
                self.node_neighbors_map[self.node_map[ link[0] ]].add(self.node_map[ link[1] ])

    def read_test_link(self):
        f = open(self.testlinkfile)
        line = f.readline()
        while line:
            line = line.strip().split(' ')
            self.X_test.append([self.node_map[ int(line[0]) ] ,  self.node_map[ int(line[1]) ] ,  int(line[2])  ] )
            line = f.readline()
        f.close()
        print("test link number:", len(self.X_test))

