#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops.functional_ops import map_fn

from romanesco.const import *


def define_computation_graph(vocab_size: int, batch_size: int):

    # Placeholders for input and output
    inputs = tf.placeholder(tf.int32, shape=(batch_size, NUM_STEPS), name='x')  # (time, batch)
    targets = tf.placeholder(tf.int32, shape=(batch_size, NUM_STEPS), name='y') # (time, batch)

    with tf.name_scope('Embedding'):
        embedding = tf.get_variable('word_embedding', [vocab_size, HIDDEN_SIZE])
        input_embeddings = tf.nn.embedding_lookup(embedding, inputs)

    with tf.name_scope('RNN'):
        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, input_embeddings, initial_state=initial_state)

    with tf.name_scope('Hidden_layer'):
        # w1: weights from RNN output to hidden layer. HIDDEN_SIZE:
        # size of RNN-output vector. SECOND_HIDDEN_SIZE: size of hidden-layer
        # vector. Sizes defined in const.py. The size of the hidden layer is
        # usually smaller thanthe size of the RNN.
        w1 = tf.get_variable('w1', shape=(HIDDEN_SIZE, SECOND_HIDDEN_SIZE))
        # b1: bias vector. One bias for each neurone of hidden layer. It has
        # the same size as the hidden layer.
        b1 = tf.get_variable('b1', SECOND_HIDDEN_SIZE)
        middle_projection = lambda x: tf.matmul(x, w1) + b1
        # Output of hidden layer: uses the middle_projection algorithm and
        # applies it to the output of the RNN.
        hidden_outputs = map_fn(middle_projection, rnn_outputs)

    with tf.name_scope('Final_Projection'):
        w2 = tf.get_variable('w2', shape=(SECOND_HIDDEN_SIZE, vocab_size))
        b2 = tf.get_variable('b2', vocab_size)
        final_projection = lambda x: tf.matmul(x, w2) + b2
        logits = map_fn(final_projection, hidden_outputs)

    with tf.name_scope('Cost'):
        # weighted average cross-entropy (log-perplexity) per symbol
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=targets,
                                                weights=tf.ones([batch_size, NUM_STEPS]),
                                                average_across_timesteps=True,
                                                average_across_batch=True)

    with tf.name_scope('Optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Logging of cost scalar (@tensorboard)
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()

    return inputs, targets, loss, train_step, logits, summary
