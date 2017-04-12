#!/usr/bin/env python3

import tensorflow as tf
import os
import argparse
from model import MeshCNN
from data_prep import sent_embedding


"""
restore most recently trained model, and use it to make an inference.
A user will need to provide a blob of sentences. Given the text, it will 
print out the scores of each document class (diagnosis, test, treatment, 
and others) along with the classified label.
"""
# Data loading params
file_test = "data/pdat.test"

# data loading
sequence_len = 58             # max number of sentences in a document

# model hyperparameters
embedding_dim = 200           # word embedding size (using pre-trained word2vec)
filter_sizes = "3,4,5"        # comma-separated filter sizes
num_filters = 128             # number of filters per filter size
dropout_keep_prob = 0.5       # dropout keep probability
l2_reg_lambda = 0.0           # l2 regularization lambda (optional)

# Training parameters
batch_size = 32               # batch size (default: 64)
num_epochs = 600              # Number of training epochs, 200
evaluate_every = 200          # Evaluate model on dev set after this many steps
checkpoint_every = 1000       # Save model after this many steps (default: 100)
num_checkpoints = 5           # number of checkpoints to store (default: 5)


def app_run(chk_path):
    with tf.Graph().as_default():
        # define a session
        session_conf = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # instantiate MeshCNN model
            cnn = MeshCNN(
                sequence_length=sequence_len,
                num_classes=4,
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(','))),
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda
            )

            # restore session with checkpoint data
            if not os.path.exists(chk_path + '.meta'):
                SystemError("checkpoint path does not exist [{}]".format(chk_path))

            print("=== restoring a model ===")
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, chk_path)
            print("model restored [{}]".format(chk_path))

            print("=== inference ===")

            while True:
                print("type senteneces ending with ctrl+d, to quit just ctrl+d: \n")
                user_input = []
                while True:
                    try:
                        line = input()
                    except EOFError:
                        break
                    if len(line) > 0:
                        user_input.append(line)
                if len(user_input) == 0:
                    break
                print("user_input size:", len(user_input))
                text = '\n'.join(user_input)
                emb = sent_embedding(text, sequence_len, embedding_dim)
                print(emb)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                scores, predict = sess.run([cnn.scores, cnn.predictions],
                                           {cnn.input_x: [emb],
                                            cnn.input_y: [[0,0,0,0]],
                                            cnn.dropout_keep_prob: 1.0})
                print(scores, predict)

            # # get accuracy on test data
            # print("=== test ===")
            #
            # # Initialize all variables
            # sess.run(tf.global_variables_initializer())
            #
            # x_text, y = load_data_and_labels("data/pdata.test")
            # cnt = 20
            # y_hat = list()
            # for i, doc in enumerate(x_text):
            #     sentences = [clean_str(s) for s in sent_tokenize(doc)]
            #     sentences_emb = np.array(list(vocab_processor.transform(sentences)))
            #     scores, predictions = \
            #         sess.run([cnn.scores, cnn.predictions],
            #                  {cnn.input_x: sentences_emb,
            #                   cnn.input_y: [y[i]] * len(sentences_emb),
            #                   cnn.dropout_keep_prob: 1.0})
            #
            #     sum = scores.sum(axis=0)
            #     y_hat.append(np.argmax(sum))
            #     print(scores, predictions)
            #     print(sum, np.argmax(sum), y[i])
            #     # exit just to see the first
            #     cnt -= 1
            #     if cnt <= 0:
            #         accuracy = sum(1 for a, b in zip(y[:i+1].tolist(), y_hat)
            #                        if b[a] == 1)
            #         print("Accuracy: {:.4f}".format(accuracy))
            #         raise SystemError()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name',
                        help="relative path to checkpoint file,\n " +
                        "ex. ./runs/1490917391/checkpoints/model-9200")
    args = parser.parse_args()

    app_run(args.file_name)

