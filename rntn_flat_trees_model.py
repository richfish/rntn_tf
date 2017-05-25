from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections

import pdb


class RNTN_alt_model(object):

    def __init__(self, input_dim, output_dim, vocab_size, l2_factor):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.l2_factor = l2_factor

    def build_core_objs(self):

        with tf.variable_scope("RNTN_main"):
            tf.get_variable("W", [self.input_dim, 2*self.input_dim])
            tf.get_variable("b", [self.input_dim, 1], initializer=tf.constant_initializer(0.0))
            tf.get_variable("V", [self.input_dim, 2*self.input_dim, 2*self.input_dim])
        with tf.variable_scope("Softmax"):
            tf.get_variable("Ws", [self.input_dim, self.output_dim])
            tf.get_variable('bs', [1, self.output_dim], initializer=tf.constant_initializer(0.0))
        with tf.variable_scope('Embeddings'):
            self.embeddings = tf.get_variable('embeddings', [self.vocab_size, self.input_dim]) # keep trainable=True

        self.lr = tf.Variable(0.0, trainable=False)

        # placeholders this version
        self.words = tf.placeholder(tf.int32, [None])
        self.labels = tf.placeholder(tf.int32, [None])
        self.l_children = tf.placeholder(tf.int32, [None])
        self.r_children = tf.placeholder(tf.int32, [None])
        self.leaves = tf.placeholder(tf.bool, [None])

        # unlike first model add to initial build
        self.train_tree()

    def train_tree(self):
        """
        core ops
        """

        # TODO adjust for root-only stats vs. all-node stats

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

        # expose for root sentiment task
        self.root_logit = self.tree_logits[-1]

        self.labels = tf.reshape(self.labels, [-1, 5])


        # debugging ln(0), custom x-entropy
        self.tree_logits = self.tree_logits + tf.constant(0.00001)
        # softmax = tf.nn.softmax(logits)
        # cross_entropy = -tf.reduce_sum(self.labels * tf.log(softmax), reduction_indices=[1])

        loss_raw = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.tree_logits, labels=self.labels))

        # per paper, include (V, W, Ws, L) l2 reg
        l2_terms = (tf.nn.l2_loss(W) + tf.nn.l2_loss(V) + tf.nn.l2_loss(Ws))
        self.tree_loss = loss_raw + self.l2_factor * l2_terms

        self.tree_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.tree_logits, 1),
            tf.argmax(self.labels, 1)), tf.float32))

        self.root_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.root_logit,
            labels=self.labels[-1])

        self.root_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.root_logit),
            tf.argmax(self.labels[-1])), tf.float32))

        self.update = self._get_optimizer().minimize(self.tree_loss)

        # nans, debug grads
        # #params = tf.trainable_variables()
        # #grads = tf.gradients(self.tree_loss, params)
        # opt = self._get_optimizer()
        # grads = opt.compute_gradients(self.tree_loss)
        # #grads, norm = tf.clip_by_global_norm(grads, 4.5) # max gradient norm, tune
        # grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads] # tune
        # self.update = opt.apply_gradients(grads)


    def composition_loop(self, t_array, i):
        # https://www.tensorflow.org/api_docs/python/tf/gather
        word = tf.gather(self.words, i)
        l_child = tf.gather(self.l_children, i)
        r_child = tf.gather(self.r_children, i)
        is_leaf = tf.gather(self.leaves, i)
        node_tensor = tf.cond( # embed or compose
            is_leaf,
            lambda: self._dynamic_embed(word),
            lambda: self._compose(t_array.read(l_child), t_array.read(r_child))
        )
        t_array = t_array.write(i, node_tensor)
        return t_array, i+1


    def _compose(self, left, right):
        """ left and right are tensors """
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

    def log_tensorboard(self):
        # TODO expand
        tf.summary.scalar("loss", self.tree_Loss)
        tf.summary.scalar("accuracy", self.tree_accuracy)
        self.merged_summary = tf.summary.merge_all()

    def _dynamic_embed(self, word):
        """ word: vocab-index int """
        with tf.variable_scope('Embeddings', reuse=True):
            embeddings = tf.get_variable('embeddings')
        return tf.transpose(
            tf.expand_dims(tf.nn.embedding_lookup(embeddings, word), 0))

    def _get_optimizer(self):
        #return tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        # Adam for static graph, woot!
        return tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-8)
