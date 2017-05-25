from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections

import pdb

# Experimenting with mini batching & padded input

class RNTN_alt_batch_model(object):

    def __init__(self, input_dim, output_dim, vocab_size, l2_factor, lr, batch_size):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.l2_factor = l2_factor
        self.lr = lr
        self.batch_size = batch_size

    def build_core_objs(self):

        with tf.variable_scope("RNTN_main"):
            # shouldnt need batch dim
            tf.get_variable("W", [self.input_dim, 2*self.input_dim])
            tf.get_variable("b", [self.input_dim, 1], initializer=tf.constant_initializer(0.0))
            tf.get_variable("V", [self.input_dim, 2*self.input_dim, 2*self.input_dim])

        with tf.variable_scope("Softmax"):
            tf.get_variable("Ws", [self.input_dim, self.output_dim])
            tf.get_variable('bs', [1, self.output_dim], initializer=tf.constant_initializer(0.0))

        with tf.variable_scope('Embeddings'):
            self.embeddings = tf.get_variable('embeddings', [self.vocab_size, self.input_dim]) # keep trainable=True

        self.lr = tf.Variable(0.0, trainable=False)

        # or [None, None] to infer both
        self.words = tf.placeholder(tf.int32, [self.batch_size, None])
        self.labels = tf.placeholder(tf.int32, [self.batch_size, None])
        self.l_children = tf.placeholder(tf.int32, [self.batch_size, None])
        self.r_children = tf.placeholder(tf.int32, [self.batch_size, None])
        self.leaves = tf.placeholder(tf.bool, [self.batch_size, None])

        self.train_batch()

    def train_batch(self):
        """
        core ops
        """

        with tf.variable_scope("Softmax", reuse=True):
            Ws = tf.get_variable("Ws")
            bs = tf.get_variable('bs')
        with tf.variable_scope("RNTN_main", reuse=True):
            W = tf.get_variable("W")
            V = tf.get_variable("V")

        # https://www.tensorflow.org/api_docs/python/tf/TensorArray
        t_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False,
                                infer_shape=False)

        # https://www.tensorflow.org/api_docs/python/tf/while_loop
        self.t_array_final, _ = tf.while_loop(lambda _, i: i < tf.shape(self.leaves)[0],
                                            self.composition_loop, [t_array, 0])
        # (all_values, 1) -> (num_nodes, input_dim)
        self.nodes_rep_matrix = tf.reshape(self.t_array_final.concat(),
                                            [-1, self.input_dim])

        self.tree_logits = tf.matmul(self.nodes_rep_matrix, Ws) + bs

        self.labels = tf.reshape(self.labels, [-1, 5])

        loss_raw = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.tree_logits, labels=self.labels))

        # per paper, include (V, W, Ws, L) l2 reg
        l2_terms = (tf.nn.l2_loss(W) + tf.nn.l2_loss(V) + tf.nn.l2_loss(Ws))
        self.tree_loss = loss_raw + self.l2_factor * l2_terms

        self.tree_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.tree_logits, 1),
            tf.argmax(self.labels, 1)), tf.float32))

        self.update = self._get_optimizer().minimize(self.tree_loss)
        # gradient checking
        # opt = self._get_optimizer()
        # params = tf.trainable_variables()
        # gradients = tf.gradients(self.loss, params)
        # self.update = opt.apply_gradients(zip(self.grads, params))


    def composition_loop(self, t_array, i):
        # https://github.com/tensorflow/tensorflow/issues/206
        # can tf.gather the transpose to iterate batch
        word = tf.gather(tf.transpose(self.words), i)
        l_child = tf.gather(tf.transpose(self.l_children), i)
        r_child = tf.gather(tf.transpose(self.r_children), i)
        is_leaf = tf.gather(tf.transpose(self.leaves), i)

        # https://www.tensorflow.org/api_docs/python/tf/where
        # NOTE blocker t_array.read for multi dims
        node_tensor = tf.where( # embed or compose
            is_leaf,
            lambda: self._dynamic_embed(word),
            lambda: self._compose(t_array.read(l_child), t_array.read(r_child))
        )
        t_array = t_array.write(i, node_tensor)
        return t_array, i+1


    def _compose(self, left, right):
        """ left and right are tensors """
        pdb.set_trace()
        h = tf.concat(axis=0, values=[left, right])

        with tf.variable_scope("RNTN_main", reuse=True):
            W = tf.get_variable("W")
            b = tf.get_variable("b")
            V = tf.get_variable("V")

        # main neural tensor action
        # or tf.tensordot(V, h, axes=1), https://www.tensorflow.org/api_docs/python/tf/tensordot
        main_rntn_tmp = tf.matmul(tf.transpose(h), tf.reshape(V, [100, 100*50]))
        main_rntn_ret = tf.matmul(tf.reshape(main_rntn_tmp, [50,100]), h)

        composed = main_rntn_ret + tf.matmul(W, h) + b
        return tf.nn.relu(composed)


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
                tf.expand_dims(tf.nn.embedding_lookup(embeddings, word), 0)
        )

    def _get_optimizer(self):
        return tf.train.GradientDescentOptimizer(learning_rate=self.lr)
