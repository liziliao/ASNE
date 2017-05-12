'''
Tensorflow implementation of Social Network Embedding framework (SNE)

@author: Lizi Liao (liaolizi.llz@gmail.com)

@references:
https://github.com/wangz10/tensorflow-playground/blob/master/word2vec.py#L105
https://www.kaggle.com/c/word2vec-nlp-tutorial/details/
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
https://github.com/wangz10/UdacityDeepLearning/blob/master/5_word2vec.ipynb
'''


import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
import evaluation

class SNE(BaseEstimator, TransformerMixin):
    def __init__(self, data, id_embedding_size, attr_embedding_size,
                 batch_size=128, alpha = 1.0, n_neg_samples=10,
                epoch=20, random_seed = 2016):
        # bind params to class
        self.batch_size = batch_size
        self.node_N = data.id_N
        self.attr_M = data.attr_M
        self.X_train = data.X
        self.X_test = data.X_test
        self.nodes = data.nodes
        self.id_embedding_size = id_embedding_size
        self.attr_embedding_size = attr_embedding_size
        self.alpha = alpha
        self.n_neg_samples = n_neg_samples
        self.epoch = epoch
        self.random_seed = random_seed
        # init all variables in a tensorflow graph
        self._init_graph()


    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():#, tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_data_id = tf.placeholder(tf.int32, shape=[None])  # batch_size * 1
            self.train_data_attr = tf.placeholder(tf.float32, shape=[None, self.attr_M])  # batch_size * attr_M
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])  # batch_size * 1

            # Variables.
            network_weights = self._initialize_weights()
            self.weights = network_weights

            # Model.
            # Look up embeddings for node_id.
            self.id_embed =  tf.nn.embedding_lookup(self.weights['in_embeddings'], self.train_data_id) # batch_size * id_dim
            self.attr_embed =  tf.matmul(self.train_data_attr, self.weights['attr_embeddings'])  # batch_size * attr_dim
            self.embed_layer = tf.concat(1, [self.id_embed, self.alpha * self.attr_embed]) # batch_size * (id_dim + attr_dim)

            ## can add hidden_layers component here!

            # Compute the loss, using a sample of the negative labels each time.
            self.loss =  tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights['out_embeddings'], self.weights['biases'], self.embed_layer,
                                                  self.train_labels, self.n_neg_samples, self.node_N))
            # Optimizer.
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            # print("AdamOptimizer")

            # init
            init = tf.initialize_all_variables()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['in_embeddings'] = tf.Variable(tf.random_uniform([self.node_N, self.id_embedding_size], -1.0, 1.0))  # id_N * id_dim
        all_weights['attr_embeddings'] = tf.Variable(tf.random_uniform([self.attr_M,self.attr_embedding_size], -1.0, 1.0)) # attr_M * attr_dim
        all_weights['out_embeddings'] = tf.Variable(tf.truncated_normal([self.node_N, self.id_embedding_size + self.attr_embedding_size],
                                    stddev=1.0 / math.sqrt(self.id_embedding_size + self.attr_embedding_size)))
        all_weights['biases'] = tf.Variable(tf.zeros([self.node_N]))
        return all_weights

    def partial_fit(self, X): # fit a batch
        feed_dict = {self.train_data_id: X['batch_data_id'], self.train_data_attr: X['batch_data_attr'],
                     self.train_labels: X['batch_data_label']}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]

    def train(self): # fit a dataset

        print 'Using in + out embedding'

        for epoch in range( self.epoch ):
            total_batch = int( len(self.X_train['data_id_list']) / self.batch_size)
            # print('total_batch in 1 epoch: ', total_batch)
            # Loop over all batches
            for i in range(total_batch):
                # generate a batch data
                batch_xs = {}
                start_index = np.random.randint(0, len(self.X_train['data_id_list']) - self.batch_size)
                batch_xs['batch_data_id'] = self.X_train['data_id_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_attr'] = self.X_train['data_attr_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_label'] = self.X_train['data_label_list'][start_index:(start_index + self.batch_size)]

                # Fit training using batch data
                cost = self.partial_fit(batch_xs)

            # Display logs per epoch
            Embeddings_out = self.getEmbedding('out_embedding', self.nodes)
            Embeddings_in = self.getEmbedding('embed_layer', self.nodes)
            Embeddings = Embeddings_out + Embeddings_in

            # link prediction test
            roc = evaluation.evaluate_ROC(self.X_test, Embeddings)
            print "Epoch:", '%04d' % (epoch + 1), \
                         "roc=", "{:.9f}".format(roc)


    def getEmbedding(self, type, nodes):
        if type == 'embed_layer':
            feed_dict = {self.train_data_id: nodes['node_id'], self.train_data_attr: nodes['node_attr']}
            Embedding = self.sess.run(self.embed_layer, feed_dict=feed_dict)
            return Embedding
        if type == 'out_embedding':
            Embedding = self.sess.run(self.weights['out_embeddings'])
            return Embedding  # nodes_number * (id_dim + attr_dim)

