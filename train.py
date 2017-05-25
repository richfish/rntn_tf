from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import numpy as np
import tensorflow as tf
import os
import time
import sys
import csv

import pdb

from rntn_model import RNTN_model
from rntn_flat_trees_model import RNTN_alt_model
from rntn_flat_trees_batch_model import RNTN_alt_batch_model
from sst_client import SSTClient, Trees


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("assets_dir", "./assets/", "GloVe, vocab-index map, etc")
tf.app.flags.DEFINE_string("sst_dir", "./sst/", "Your SST data")
tf.app.flags.DEFINE_string("ckpt_dir", "./checkpoints/", "Model checkpoints")
tf.app.flags.DEFINE_boolean("mini", False, "Use tiny subset of data .e.g. debug, iterate ")
tf.app.flags.DEFINE_boolean("train_flat", True, "Recommended, flat trees/ static graph approach.")
tf.app.flags.DEFINE_boolean("batch_flat", False, "*** unfinished *** batch, pad, bucket input for mini batches.")
tf.app.flags.DEFINE_boolean("test", False, "Set True to run test trees.")
tf.app.flags.DEFINE_integer("vocab_size", 19540, "Vocab of SST corpus + internal tokens")
tf.app.flags.DEFINE_integer("input_dim", 50, "E.g. GloVe dimension of input")
tf.app.flags.DEFINE_integer("output_dim", 5, "SST labels")
tf.app.flags.DEFINE_integer("num_epochs", 50, "Max number of epochs")
tf.app.flags.DEFINE_integer("check_in_every", 20, "check-in for stats, lr every n trees")
tf.app.flags.DEFINE_integer("save_every", 400, "save every n trees")
tf.app.flags.DEFINE_integer("batch_size", 10, "*** unfinished *** batch size for flat trees")
tf.app.flags.DEFINE_float("l2_factor", 0.02, "Do l2 regularization per paper")
tf.app.flags.DEFINE_float("lr", 1e-3, "Fyi manage own lr for tree input")
tf.app.flags.DEFINE_float("lr_decay", 0.99, "Learning rate decay for optimizer.")


sst_client = SSTClient(FLAGS.assets_dir, FLAGS.sst_dir)

if not os.path.exists(FLAGS.assets_dir + "precomputed_50d_glove.weights.npy"):
    print("compiling GloVe for TF embeddings")
    sst_client.compile_glove()

embedding_matrix = np.load(os.path.join(FLAGS.assets_dir, "precomputed_50d_glove.weights.npy"))

tf.logging.set_verbosity(tf.logging.INFO)



def train():

    train_trees, dev_trees, _ = Trees(FLAGS.assets_dir, FLAGS.sst_dir, FLAGS.mini) \
        .load_train_dev_test_trees()

    if FLAGS.train_flat:
        print("preprocessing flat trees\n")
        train_trees = Trees().preprocess_flat(train_trees)
        dev_trees = Trees().preprocess_flat(dev_trees)

    if FLAGS.batch_flat:
        print("batching & bucketing flat trees")
        train_trees = Trees().batch_and_bucket(train_trees)

    with tf.Graph().as_default(), tf.Session() as session:

        model = create_model()

        model.build_core_objs()

        session.run(tf.global_variables_initializer())

        #model.assign_embeddings(session, embedding_matrix) # tmp debug
        model.assign_lr(session, FLAGS.lr)

        saver = save_or_restore(session)

        #summary_writer = tf.summary.FileWriter(FLAGS.tensorboard, graph=tf.get_default_graph())
        start_time = time.time()
        val_running_acc = []

        for epoch_i in range(1,FLAGS.num_epochs):

            epoch_loss = []; epoch_acc = []
            step_loss = []; step_acc = []

            print("\nEPOCH {:d}\n".format(epoch_i))

            for i,tree in enumerate(train_trees): # in order for debug, easy lookup in dataset

                if not FLAGS.train_flat:
                    model.train_tree(tree)

                feed = {}
                if FLAGS.train_flat:
                    feed = { model.words: tree[0], model.labels: tree[1], model.l_children: tree[2],
                            model.r_children: tree[3], model.leaves: tree[4] }

                fetches = [model.tree_logits, model.nodes_rep_matrix, model.tree_loss,
                            model.tree_accuracy, model.update]
                tree_logits, rep_matrix, loss, accuracy, _ = session.run(fetches, feed)

                if np.isnan(loss):
                    pdb.set_trace()
                    raise Exception("nan alert - stopping.")

                step_loss.append(loss); step_acc.append(accuracy)

                if i % FLAGS.check_in_every == 0 and i != 0:
                    avg_acc, avg_loss = np.mean(step_acc), np.mean(step_loss)

                    print("TRAIN step {:d} epoch {:d} after {:.2f}s ---> accuracy {:.6f}, loss {:6f}\n" \
                            .format(i, epoch_i, time.time() - start_time, avg_acc, avg_loss))

                    if len(step_loss) > 4 and loss > max(step_loss[-4:]):
                        model.decay_lr(session, FLAGS.lr_decay_factor)

                    epoch_loss.append(avg_loss); epoch_acc.append(avg_acc)
                    step_loss = []; step_acc = []

                if (i * epoch_i) % FLAGS.save_every == 0 and i != 0:
                    if not os.path.exists(FLAGS.ckpt_dir): os.mkdir(FLAGS.ckpt_dir)

                    print("running dev batch on root sentiment")
                    root_accs = []; root_loss = []; root_logits = []
                    for _ in xrange(200): # pick default
                        tree = dev_trees[np.random.randint(0,len(dev_trees)-1)]
                        if FLAGS.train_flat:
                            feed = { model.words: tree[0], model.labels: tree[1], model.l_children: tree[2],
                                    model.r_children: tree[3], model.leaves: tree[4] }
                        fetches = [model.root_acc, model.root_logit, model.root_loss]
                        acc, logit, loss = session.run(fetches, feed)
                        if np.isnan(loss):
                            continue
                        root_accs.append(acc); root_logits.append(logit); root_loss.append(loss)

                    print("DEV root accuracy {:.6f}, loss {:.6f}\n".format(np.mean(root_accs),
                            np.mean(root_loss)))

                    val_running_acc.append(np.mean(root_accs))
                    if epoch_i > 5 and val_running_acc[-1:] < min(val_running_acc[-10:-1]):
                        continue # don't save if val dropping off

                    saver.save(session, FLAGS.ckpt_dir + "rntn_flat_tree_model",
                        global_step= i * epoch_i)

            print("TRAIN epoch after: {:.2f}s ---> accuracy: {:.6f}, loss: {:.6f}\n".format(
                time.time() - start_time, np.average(epoch_acc[-20:]), np.average(epoch_loss[-20:])))



def create_model():
    core = [
        FLAGS.input_dim,
        FLAGS.output_dim,
        FLAGS.vocab_size,
        FLAGS.l2_factor]
    if FLAGS.batch_flat:
        model = RNTN_alt_batch_model(*core + [FLAGS.batch_size])
    elif FLAGS.train_flat:
        model = RNTN_alt_model(*core)
    else:
        model = RNTN_model(*core)
    return model


def save_or_restore(sess):
    saver = tf.train.Saver(max_to_keep=5)
    if os.path.exists(FLAGS.ckpt_dir) and os.listdir(FLAGS.ckpt_dir) != []:
        print("\nrestoring model parameters\n")
        # if want to import graph
        # saver = tf.train.import_meta_graph('rntn_flat_tree_model-{:d}.meta'.format(step))
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
    else:
        print("\ncreating model with fresh parameters.\n")
    return saver


def test():
    test_trees = Trees(FLAGS.assets_dir, FLAGS.sst_dir, FLAGS.mini, mode='test') \
        .load_trees()

    if FLAGS.train_flat:
        print("preprocessing flat trees for test\n")
        test_trees_flat = Trees().preprocess_flat(test_trees)

    with tf.Graph().as_default(), tf.Session() as session:
        model = create_model()
        model.build_core_objs()
        session.run(tf.global_variables_initializer())
        model.assign_embeddings(session, embedding_matrix)
        saver = save_or_restore(session)
        idx_guess_label = []
        for i,tree in enumerate(test_trees_flat):
            feed = {}
            if FLAGS.train_flat:
                feed = { model.words: tree[0], model.labels: tree[1], model.l_children: tree[2],
                        model.r_children: tree[3], model.leaves: tree[4] }

            acc, logit, all_node_acc = session.run([model.root_acc, model.root_logit, model.tree_accuracy], feed)

            label = test_trees[i].root.label
            idx_guess_label.append((i, logit, label, np.argmax(logit), np.argmax(label), all_node_acc))
        print("writing results to tasks/rntn_predictions_raw.csv")
        with open("./tasks/rntn_predictions_raw.csv", "w") as f:
            writer = csv.writer(f)
            # NOTE if training on coarse root labels tweak this
            writer.writerow(["test_id", "logit_raw_root", "y_onehot_root", "fine_guess", "fine_label", "all_node_acc"])
            writer.writerows(idx_guess_label)




def main(_):
  if FLAGS.test:
    test()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
