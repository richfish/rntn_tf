from collections import defaultdict, OrderedDict
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

import numpy as np
import os
import cPickle
import copy
import pdb

_UNK = "unk"

class SSTClient():

    def __init__(self, assets_dir="./assets/", sst_dir="./sst/", glove_dim=50):
        self.assets_dir = assets_dir
        self.sst_dir = sst_dir
        self.glove_dim = glove_dim

    def compile_glove(self):
        """
        compile GloVe matrix based on SST vocab, to replace TF embedding_lookup
        https://nlp.stanford.edu/projects/glove/ using 6B.50d
        nb: 300d at tensor layer (600^2 * 300) vs. 50d (100^2 * 50)        """

        if not os.path.exists(self.assets_dir + 'glove.6B.50d.txt'):
            raise Exception("glove.6B.50d.txt download not in assets_dir")

        # prepare & save basic vocab-index mappings
        tokenizer = Tokenizer(lower=True, filters='')
        tokens_file = os.path.join(self.sst_dir, "SOStr.txt")
        with open(tokens_file, 'r') as myfile:
            txt_raw = myfile.read()
        words = []
        for sent in txt_raw.split("\n"):
            words.extend(sent.split("|"))
        tokenizer.fit_on_texts(words)
        vocab_index_map = tokenizer.word_index
        vocab_index_map[_UNK] = len(vocab_index_map)+1
        if not os.path.exists(self.assets_dir + 'vocab_index_map.bin'):
            with open(os.path.join(self.assets_dir,'vocab_index_map.bin'),'w') as f:
                cPickle.dump(vocab_index_map, f)

        self.glove_store = self.assets_dir + "/precomputed_50d_glove.weights"
        if not os.path.exists(self.glove_store + '.npy'):
            embeddings_index = {}
            glove_path = os.path.join(self.assets_dir, "glove.6B.50d.txt")
            f = open(glove_path)
            for line in f:
                values = line.split(' ')
                word = values[0]
                features = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = features
            f.close()

            vocab_size = len(vocab_index_map)
            embedding_matrix = np.zeros((vocab_size+1, self.glove_dim))
            for i, word in vocab_index_map.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is None:
                    embedding_matrix[i] = embeddings_index[_UNK]
                else:
                    embedding_matrix[i] = embedding_vector
            print("matrix done - size {}".format(len(embedding_matrix)))
            np.save(self.glove_store, embedding_matrix)


class Trees:
    def __init__(self, assets_dir="./assets/", sst_dir="./sst/", mini=False,
                mode='train'):
        self.assets_dir = assets_dir
        self.sst_dir = sst_dir
        self.mode = mode
        self.trees_path = os.path.join(self.sst_dir, mode + ".txt")
        self.mini = mini

        with open(os.path.join(self.assets_dir, "vocab_index_map.bin"),'r') as f:
            self.vocab_index_map = cPickle.load(f)

    def load_trees(self, trees_path=None):
        trees_path = trees_path or self.trees_path
        with open(trees_path, 'r') as f:
            trees = [Tree(l) for l in f.readlines()]
        for tree in trees:
            self._map_vocab_for_tree(tree.root)
        return trees

    def load_train_dev_test_trees(self):
        if self.mini:
            trees_path = os.path.join(self.sst_dir, "mini.txt")
            mini = self.load_trees(trees_path)
            return [mini, mini, mini]
        all_sets = []
        for mode in ['train', 'dev', 'test']:
            trees_path = os.path.join(self.sst_dir, mode + ".txt")
            all_sets.append(self.load_trees(trees_path))
        return all_sets

    def preprocess_flat(self, trees):
        """ returns shape (num_trees, 5, num_nodes) """
        return [t.flat_representation() for t in trees]


    def batch_and_bucket(self, trees):
        """
        takes input from #preprocess_flat
        idea: let padding be leaves and don't compose
        prepend padding so can still root test in batch
        monitor how impacts accuracy but should learn to ignore
        """
        # NOTE work in progress
        # TODO move config after working
        # TODO bucket by comparable lengths
        maxlen = 60
        bactch_size = 10
        batch = [[], [], [], [], []]
        padded_trees = []; all_batches = []
        for tree in trees:
            if len(tree[0]) >= maxlen:
                words = tree[0][:maxlen]
                labels = tree[1][:maxlen]
                l_children = tree[2][:maxlen]
                r_children = tree[3][:maxlen]
                leaves = tree[4][:maxlen]
            else:
                diff = maxlen - len(tree)
                words = [_UNK]*diff + tree[0]
                labels = [0,0,0,0,1]*diff + tree[1]
                l_children = [-1]*diff + tree[2]
                r_children = [-1]*diff + tree[3]
                leaves = [True]*diff + tree[4]
            padded_trees.append([words,labels,l_children,r_children,leaves])

        for i in range(0, len(trees), bactch_size):
            batch = copy.deep_copy(batch)
            for tree in trees[i:i+batch_size]:
                batch[0].append(tree[0])
                batch[1].append(tree[1])
                batch[2].append(tree[2])
                batch[3].append(tree[3])
                batch[4].append(tree[4])
        return all_batches


    def _map_vocab_for_tree(self, node):
        if node.isLeaf:
            if node.word not in self.vocab_index_map:
                node.word = self.vocab_index_map[_UNK]
            else:
                node.word = self.vocab_index_map[node.word]
        if node.left is not None:
            self._map_vocab_for_tree(node.left)
        if node.right is not None:
            self._map_vocab_for_tree(node.right)


class Tree:
    def __init__(self, tree_raw):
        tokens = []
        for toks in tree_raw.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)
        self.labels = self._get_labels(self.root)

    def flat_representation(self):
        """
        assumes parsed
        outputs words, labels, l_children, r_children each of length total nodes
        """
        postorder_nodes = []; stack = [(self.root, False)]
        while stack:
            node, visited = stack.pop()
            if node:
                if visited:
                    postorder_nodes.append(node)
                else:
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))

        indices = OrderedDict(map(reversed, enumerate(postorder_nodes)))
        words, labels, l_children, r_children, leaves = [],[],[],[],[]
        for node in postorder_nodes:
            words.append(node.word or -1)
            labels.append(node.label)
            l_children.append(indices[node.left] if not node.isLeaf else -1)
            r_children.append(indices[node.right] if not node.isLeaf else -1)
            leaves.append(node.isLeaf)
        return words, labels, l_children, r_children, leaves

    def parse(self, tokens, parent=None):
        """ string chunks to node objs """
        split = 2 # position after open and label
        countOpen = countClose = 0
        if tokens[split] == "(":
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == "(":
                countOpen += 1
            if tokens[split] == ")":
                countClose += 1
            split += 1
        # new node with one hot senti label
        label = np_utils.to_categorical(int(tokens[1]), 5).tolist()[0]
        node = Node(label)
        node.parent = parent
        # leaf node
        if countOpen == 0:
            node.word = ''.join(tokens[2:-1]).lower() # lower case?
            node.isLeaf = True
            return node
        node.left = self.parse(tokens[2:split],parent=node)
        node.right = self.parse(tokens[split:-1],parent=node)
        return node

    def _get_labels(self, node):
        if node is None: return []
        return self._get_labels(node.left) + self._get_labels(node.right) + \
            node.label

class Node:
    def __init__(self, label, word=None):
        """
        label: one-hot vector
        h: activations of node after composition ops
        """
        self.label = label
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False
        self.h = None
        self.logits = None # optionally assign these for alt implementation
        self.loss = None # ibid
