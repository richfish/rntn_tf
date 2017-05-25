from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections

import pdb

# TODO TF Fold for dynamic batching over variable structured trees https://github.com/tensorflow/fold

class RNTN_model(object):

    def __init__(self, input_dim, output_dim, vocab_size, l2_factor, lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.l2_factor = l2_factor
        self.lr = lr

    def build_core_objs(self):
        # can do without x,y placeholders
        # self.X = tf.placeholder(tf.int32, [None, 1]) # raw leaf value
        # self.y = tf.placeholder(tf.float32, [None, self.output_dim], name='targets_y') # label per leaf
        with tf.variable_scope("RNTN_main"):
            tf.get_variable("W", [self.input_dim, 2*self.input_dim])
            tf.get_variable("b", [self.input_dim, 1], initializer=tf.constant_initializer(0.0))
            tf.get_variable("V", [self.input_dim, 2*self.input_dim, 2*self.input_dim])
        with tf.variable_scope("Softmax"):
            tf.get_variable("Ws", [self.input_dim, self.output_dim])
            tf.get_variable('bs', [1, self.output_dim], initializer=tf.constant_initializer(0.0))
        with tf.variable_scope('Embeddings'):
            self.embeddings = tf.get_variable('embeddings', [self.vocab_size, self.input_dim]) #keep trainable=True
        #self.Leaf = tf.nn.embedding_lookup(self.W_embed_X, self.X) # move in traversal stack
        self.lr = tf.Variable(0.0, trainable=False)

    def train_tree(self, tree):
        """
        core ops
        single traversal puts all node representations in matrix
        use for downstream batched ops
        """
        with tf.variable_scope("Softmax", reuse=True):
            Ws = tf.get_variable("Ws")
            bs = tf.get_variable('bs')
        with tf.variable_scope("RNTN_main", reuse=True):
            W = tf.get_variable("W")
            V = tf.get_variable("V")

        labels = tf.reshape(tree.labels, [-1, 5])

        nodes_ret = self._build_node_representation(tree.root)
        # (num_nodes, input_dim)
        self.nodes_rep_matrix = tf.transpose(tf.concat(axis=1, values=nodes_ret.values()))

        self.tree_logits = tf.matmul(self.nodes_rep_matrix, Ws) + bs

        loss_raw = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.tree_logits, labels=labels))

        # per paper, include (V, W, Ws, L) reg
        l2_terms = (tf.nn.l2_loss(W) + tf.nn.l2_loss(V) + tf.nn.l2_loss(Ws))
        self.tree_loss = loss_raw + self.l2_factor * l2_terms

        self.tree_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.tree_logits, 1),
            tf.argmax(labels, 1)), tf.float32))

        self.update = self._get_optimizer().minimize(self.tree_loss)
        # gradient checking
        # opt = self._get_optimizer()
        # params = tf.trainable_variables()
        # gradients = tf.gradients(self.loss, params)
        # self.update = opt.apply_gradients(zip(self.grads, params))


    def _build_node_representation(self, node):
        """
        returns map of node -> composed rep
        maintain order of traversal for downstream ops
        """

        with tf.variable_scope("RNTN_main", reuse=True):
            W = tf.get_variable("W")
            b = tf.get_variable("b")
            V = tf.get_variable("V")

        full_nodes_rep = collections.OrderedDict()

        if node.isLeaf:
            node.h = self._dynamic_embed(node.word)
        else:
            full_nodes_rep.update(self._build_node_representation(node.left))
            full_nodes_rep.update(self._build_node_representation(node.right))
            h = tf.concat(axis=0, values=[node.left.h, node.right.h])

            # main neural tensor action
            # alt?: tf.tensordot(V, h, axes=1) # https://www.tensorflow.org/api_docs/python/tf/tensordot
            main_rntn_tmp = tf.matmul(tf.transpose(h), tf.reshape(V, [100, 100*50]))
            main_rntn_ret = tf.matmul(tf.reshape(main_rntn_tmp, [50,100]), h)

            composed = main_rntn_ret + tf.matmul(W, h) + b
            node.h = tf.nn.relu(composed)

        full_nodes_rep[node] = node.h
        return full_nodes_rep


    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def decay_lr(self, session, decay_rate):
        session.run(tf.assign(self.lr, self.lr * decay_rate))

    def assign_embeddings(self, session, embedding_matrix):
        session.run(self.embeddings.assign(embedding_matrix))

    def _dynamic_embed(self, word):
        """ word: vocab-index int """
        with tf.variable_scope('Embeddings', reuse=True):
            embeddings = tf.get_variable('embeddings')
        return tf.transpose(
            tf.expand_dims(tf.nn.embedding_lookup(embeddings, word), 0))

    def _get_optimizer(self):
        return tf.train.GradientDescentOptimizer(learning_rate=self.lr)
