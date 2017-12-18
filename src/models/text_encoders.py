from __future__ import print_function
from __future__ import division
import tensorflow as tf
import src.tf_utils as tf_utils
import numpy as np
import math
import json
import sys


class TextEncoder(object):
    def __init__(self, text_batch, e1_dist_batch, e2_dist_batch, seq_len_batch,
                 lstm_dim, embed_dim, position_dim, token_dim,
                 bidirectional, peephole, max_pool, word_dropout_keep, lstm_dropout_keep, final_dropout_keep,
                 entity_index=100, filterwidth=3, pos_encode_batch=None, filter_pad=False, string_int_maps=None,
                 e1_batch=None, e2_batch=None, project_inputs=False):

        self.text_batch = text_batch
        self.position_dim = position_dim
        self.e1_dist_batch = e1_dist_batch
        self.e2_dist_batch = e2_dist_batch
        self.seq_len_batch = seq_len_batch
        self.e1_batch = e1_batch
        self.e2_batch = e2_batch

        self.lstm_dim = lstm_dim
        self.embed_dim = embed_dim
        self.token_dim = token_dim
        self.position_dim = position_dim
        self.word_dropout_keep = word_dropout_keep
        self.lstm_dropout_keep = lstm_dropout_keep
        self.final_dropout_keep = final_dropout_keep
        self.entity_index = entity_index
        self.pos_encode_batch = pos_encode_batch
        self.encode_position = False
        self.project_inputs = project_inputs
        self.filter_pad = filter_pad
        self.string_int_maps = string_int_maps
        if self.string_int_maps and 'token_str_id_map' in self.string_int_maps and \
                self.string_int_maps['token_str_id_map'] and '<PAD>' in self.string_int_maps['token_str_id_map']:
            self.pad_idx = self.string_int_maps['token_str_id_map']['<PAD>']
        else:
            self.pad_idx = 0

        # if self.encode_position:
        #     max_pos = 25000
        #     pos_embed = np.array([[math.sin(pos/(10000**(2*i/self.token_dim))) if i % 2 == 0
        #                                else math.cos(pos/(10000**(2*i/self.token_dim)))
        #                                for i in range(self.token_dim+(2*self.position_dim))]
        #                           for pos in range(max_pos)])
        #     self.pos_encoding = tf.get_variable(name='pos_encoding',
        #                                         initializer=pos_embed.astype('float32'),
        #                                         trainable=False)

    def get_token_embeddings(self, token_embeddings, position_embeddings, token_attention=None):
        selected_words = tf.nn.embedding_lookup(token_embeddings, self.text_batch)
        if self.project_inputs:
            params = {"inputs": selected_words, "filters": self.embed_dim, "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            selected_words = tf.layers.conv1d(**params)
        if self.position_dim > 0:
            selected_e1_dists = tf.nn.embedding_lookup(position_embeddings, self.e1_dist_batch)
            selected_e2_dists = tf.nn.embedding_lookup(position_embeddings, self.e2_dist_batch)
            token_embeds = tf.concat(axis=2, values=[selected_words, selected_e1_dists, selected_e2_dists])
        else:
            token_embeds = selected_words

        if self.encode_position:
            pad_mask = tf.expand_dims(tf.cast(tf.not_equal(self.text_batch, self.pad_idx), tf.float32), [2])
            pos_encoding = pad_mask * tf.nn.embedding_lookup(self.pos_encoding, self.pos_encode_batch)
            token_embeds = tf.add(token_embeds, pos_encoding)

        dropped_embeddings = tf.nn.dropout(token_embeds, self.word_dropout_keep)
        # keep pad tokens as 0 vectors
        if self.filter_pad:
            print('Filtering pad tokens')
            dropped_embeddings = tf.multiply(dropped_embeddings,
                                             tf.expand_dims(tf.cast(tf.not_equal(self.text_batch, self.pad_idx),
                                                                    tf.float32), [2]))

        return dropped_embeddings
