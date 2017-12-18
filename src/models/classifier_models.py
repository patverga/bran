import tensorflow as tf
import numpy as np
import sys
from src.models.text_encoders import *
from src.models.transformer import *


class ClassifierModel(object):
    def __init__(self, ep_vocab_size, entity_vocab_size, kb_vocab_size, token_vocab_size, position_vocab_size,
                 ner_label_vocab_size, embeddings, entity_embeddings, string_int_maps, FLAGS):
        self.model_type = 'classifier'
        self._lr = FLAGS.lr
        self._embed_dim = FLAGS.embed_dim
        self.inner_embed_dim = FLAGS.embed_dim
        self._token_dim = FLAGS.token_dim
        self._lstm_dim = FLAGS.lstm_dim
        self._position_dim = FLAGS.position_dim
        self._kb_size = kb_vocab_size
        self._token_size = token_vocab_size
        self._position_size = position_vocab_size
        self._ep_vocab_size = ep_vocab_size
        self._entity_vocab_size = entity_vocab_size
        self._num_labels = FLAGS.num_classes
        self._peephole = FLAGS.use_peephole
        self.string_int_maps = string_int_maps

        self._epsilon = tf.constant(0.00001, name='epsilon')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.non_linear = tf.nn.tanh if FLAGS.use_tanh else tf.identity
        self.max_pool = FLAGS.max_pool
        self.bidirectional = FLAGS.bidirectional
        self.margin = FLAGS.margin
        self.verbose = FLAGS.verbose
        self.freeze = FLAGS.freeze
        self.mlp = FLAGS.mlp

        # set up placeholders
        self.label_batch = tf.placeholder_with_default([0], [None], name='label_batch')
        self.ner_label_batch = tf.placeholder_with_default([[0]], [None, None], name='ner_label_batch')
        self.kb_batch = tf.placeholder_with_default([0], [None], name='kb_batch')
        self.e1_batch = tf.placeholder_with_default([0], [None], name='e1_batch')
        self.e2_batch = tf.placeholder_with_default([0], [None], name='e2_batch')
        self.ep_batch = tf.placeholder_with_default([0], [None], name='ep_batch')
        self.text_batch = tf.placeholder_with_default([[0]], [None, None], name='text_batch')
        self.e1_dist_batch = tf.placeholder_with_default([[0]], [None, None], name='e1_dist_batch')
        self.e2_dist_batch = tf.placeholder_with_default([[0]], [None, None], name='e2_dist_batch')
        self.ep_dist_batch = tf.placeholder_with_default([[[0.0]]], [None, None, None], name='ep_dist_batch')

        self.pos_encode_batch = tf.placeholder_with_default([[0]], [None, None], name='pos_encode_batch')
        self.seq_len_batch = tf.placeholder_with_default([0], [None], name='seq_len_batch')
        self.loss_weight = tf.placeholder_with_default(1.0, [], name='loss_weight')
        self.example_loss_weights = tf.placeholder_with_default([1.0], [None], name='example_loss_weights')

        self.word_dropout_keep = tf.placeholder_with_default(1.0, [], name='word_keep_prob')
        self.lstm_dropout_keep = tf.placeholder_with_default(1.0, [], name='lstm_keep_prob')
        self.final_dropout_keep = tf.placeholder_with_default(1.0, [], name='final_keep_prob')
        self.noise_weight = tf.placeholder_with_default(0.0, [], name='noise_weight')
        self.k_losses = tf.placeholder_with_default(0, [], name='k_losses')
        self.text_update = tf.placeholder_with_default(True, [], name='text_update')

        # initialize embedding tables
        with tf.variable_scope('noise_classifier'):
            if embeddings is None:
                self.token_embeddings = tf.get_variable(name='token_embeddings',
                                                        shape=[self._token_size, self._token_dim],
                                                        initializer=tf.contrib.layers.xavier_initializer(),
                                                        trainable=(not FLAGS.freeze)
                                                        )
            else:
                self.token_embeddings = tf.get_variable(name='token_embeddings',
                                                        initializer=embeddings.astype('float32'),
                                                        # trainable=(not freeze)
                                                        )
            self.token_embeddings = tf.concat((tf.zeros(shape=[1, FLAGS.token_dim]),
                                               self.token_embeddings[1:, :]), 0)

            self.position_embeddings = tf.get_variable(name='position_embeddings',
                                                       shape=[self._position_size, self._position_dim],
                                                       initializer=tf.contrib.layers.xavier_initializer()
                                                       ) if self._position_dim > 0 else tf.no_op()
            # w_1 = tf.get_variable(name='attention', shape=[self.inner_embed_dim, self._num_labels],
            #                            initializer=tf.contrib.layers.xavier_initializer())
            # MLP for scoring encoded sentence
            self.w_1 = tf.get_variable(name='w_1', shape=[self.inner_embed_dim, self._num_labels],
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.b_1 = tf.get_variable(name='b_1', initializer=tf.constant(0.0001, shape=[self._num_labels]))

            skip_sentence = True
            if 'transform' in FLAGS.text_encoder:
                if 'cnn' in FLAGS.text_encoder and 'only' in FLAGS.text_encoder:
                    text_encoder_type = CNNAllPairs
                else:
                    text_encoder_type = Transformer
            elif 'glu' in FLAGS.text_encoder:
                text_encoder_type = GLUAllPairs
            else:
                print('%s is not a valid text encoder' % FLAGS.text_encoder)
                sys.exit(1)
            filter_width = int(FLAGS.text_encoder.split('_')[-1]) if FLAGS.text_encoder.split('_')[-1].isdigit() else 3
            self.text_encoder = text_encoder_type(self.text_batch, self.e1_dist_batch, self.e2_dist_batch, self.ep_dist_batch,
                                                  self.seq_len_batch,
                                                  self._lstm_dim, self._embed_dim, self._position_dim, self._token_dim,
                                                  self.bidirectional, self._peephole, self.max_pool,
                                                  self.word_dropout_keep, self.lstm_dropout_keep, self.final_dropout_keep,
                                                  layer_str=FLAGS.layer_str, pos_encode_batch=self.pos_encode_batch,
                                                  filterwidth=filter_width, block_repeats=FLAGS.block_repeats,
                                                  filter_pad=FLAGS.filter_pad, string_int_maps=string_int_maps,
                                                  entity_index=1,
                                                    # entity_index=position_vocab_size/2,
                                                  e1_batch=self.e1_batch, e2_batch=self.e2_batch,
                                                  entity_embeddings=entity_embeddings, entity_vocab_size=entity_vocab_size,
                                                  num_labels=FLAGS.num_classes, project_inputs=FLAGS.freeze
                                                  )


            # encode the batch of sentences into b x d matrix
            token_attention = None # tf.squeeze(self.get_ep_embedding(), [1])
            self.attention_vector = tf.nn.embedding_lookup(tf.transpose(self.w_1), tf.ones_like(self.e1_batch))
            encoded_text_list = self.text_encoder.embed_text(self.token_embeddings, self.position_embeddings,
                                                             self.attention_vector, token_attention=token_attention)
            no_drop_output_list = []
            if FLAGS.dropout_loss_weight > 0.0:
                no_drop_output_list = self.text_encoder.embed_text(self.token_embeddings, self.position_embeddings,
                                                                   self.attention_vector, no_dropout=True, reuse=True)
            if skip_sentence:
                predictions_list = encoded_text_list
            else:
                predictions_list = [self.score_sentence(et, reuse=(False if block_num == 0 else True))
                                    for block_num, et in enumerate(encoded_text_list)]
            predictions_list = [self.add_bias(pred) for pred in predictions_list]
            self.predictions = predictions_list[-1]
            self.probs = self.get_probs()

            self.loss = self.calculate_loss(encoded_text_list, predictions_list, self.label_batch,
                                            FLAGS.l2_weight, FLAGS.dropout_loss_weight, no_drop_output_list)
            self.accuracy = self.calculate_accuracy(self.predictions, self.label_batch)
            self.text_variance = self.negative_prob(self.predictions)
            self.text_kb = self.positive_prob(self.predictions)

            self.ner_predictions, self.ner_loss = self.ner(ner_labels=ner_label_vocab_size)

    def get_probs(self):
        return tf.nn.softmax(self.predictions)


    def get_ep_embedding(self, non_linear=tf.identity):
        self.entity_embeddings = tf.get_variable(name='entity_embeddings',
                                                 shape=[self._entity_vocab_size, self._embed_dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        e1_embeddings = tf.nn.embedding_lookup(self.entity_embeddings, self.e1_batch)
        e2_embeddings = tf.nn.embedding_lookup(self.entity_embeddings, self.e2_batch)
        pos_e1_expanded = non_linear(tf.expand_dims(e1_embeddings, 1))
        pos_e2_expanded = non_linear(tf.expand_dims(e2_embeddings, 1))
        ep_embeddings = tf.multiply(pos_e1_expanded, pos_e2_expanded)
        return ep_embeddings


    def calculate_loss(self, encoded_text_list, logits_list, labels, l2_weight,
                       dropout_weight=0.0, no_drop_output_list=None, label_smoothing=.0):
        loss = 0.0
        labels = tf.one_hot(labels, self._num_labels, on_value=1.0-label_smoothing, off_value=label_smoothing)
        for logits in logits_list:
            loss_batch = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss += tf.reduce_sum(self.example_loss_weights * loss_batch)
        # l2 loss on weights
        if l2_weight > 0.0:
            for et in encoded_text_list:
                loss += tf.constant(l2_weight) * \
                       (tf.nn.l2_loss(self.w_1) + tf.nn.l2_loss(self.w_0)
                        + tf.nn.l2_loss(et)
                        )
        # difference between output with and without dropout
        for et, no_drop_et in zip(encoded_text_list, no_drop_output_list):
            loss += dropout_weight * tf.nn.l2_loss(no_drop_et-et)
        return self.loss_weight * (loss/float(len(logits_list)))

    def calculate_accuracy(self, logits, labels):
        correct_prediction = tf.equal(labels, tf.cast(tf.argmax(logits, 1), tf.int32))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def score_sentence(self, encoded_text, reuse=False):
        with tf.variable_scope('score_sentence', reuse=reuse):
            if self.mlp:
                self.w_0 = tf.get_variable(name='w_0', shape=[self._embed_dim, self.inner_embed_dim],
                                           initializer=tf.contrib.layers.xavier_initializer())
                self.b_0 = tf.get_variable(name='b_0', initializer=tf.constant(0.0001, shape=[self.inner_embed_dim]))
                prediction_0 = tf.nn.relu(tf.nn.xw_plus_b(encoded_text, self.w_0, self.b_0))
                logits = tf.nn.xw_plus_b(prediction_0, self.w_1, self.b_1)
            else:
                logits = tf.nn.xw_plus_b(encoded_text, self.w_1, self.b_1)
            return logits

    def positive_prob(self, predictions):
        probs = tf.nn.softmax(predictions)
        return probs[:, 1]

    def negative_prob(self, predictions):
        probs = tf.nn.softmax(predictions)
        return probs[:, 0]

    def ner_predictions(self, tokens, ner_labels, reuse=False):
        # MLP for scoring encoded sentence
        with tf.variable_scope('score_sentence', reuse=reuse):
            self.ner_w_1 = tf.get_variable(name='ner_w_1', shape=[self.inner_embed_dim, ner_labels],
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.ner_b_1 = tf.get_variable(name='ner_b_1', initializer=tf.constant(0.0001, shape=[ner_labels]))
            if self.mlp:
                self.ner_w_0 = tf.get_variable(name='ner_w_0', shape=[self._embed_dim, self.inner_embed_dim],
                                               initializer=tf.contrib.layers.xavier_initializer())
                self.ner_b_0 = tf.get_variable(name='ner_b_0', initializer=tf.constant(0.0001, shape=[self.inner_embed_dim]))
                prediction_0 = tf.nn.relu(tf.nn.xw_plus_b(tokens, self.ner_w_0, self.ner_b_0))
                logits = tf.nn.xw_plus_b(prediction_0, self.ner_w_1, self.ner_b_1)
            else:
                logits = tf.nn.xw_plus_b(tokens, self.ner_w_1, self.ner_b_1)
            return logits

    def ner(self, ner_labels=6, pad_idx=0):
        encoded_tokens_list = self.text_encoder.embed_text(self.token_embeddings, self.position_embeddings, self.attention_vector,
                                                           return_tokens=True, reuse=True)
        loss = 0.0
        flat_labels = tf.reshape(self.ner_label_batch, [-1])
        self.flat_mask = tf.cast(tf.not_equal(flat_labels, tf.constant(pad_idx)), tf.float32)
        for block_num, et in enumerate(encoded_tokens_list):
            reuse = False if block_num == 0 else True
            flat_tokens = tf.reshape(et, [-1, self._embed_dim])
            flat_logits = self.ner_predictions(flat_tokens, ner_labels, reuse=reuse)
            token_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
            loss += tf.reduce_sum(tf.reduce_mean(tf.multiply(token_losses, self.flat_mask), reduction_indices=[-1]))
        return flat_logits, (self.loss_weight*(loss/float(len(encoded_tokens_list))))

    def add_bias(self, x):
        return x


class MultiLabelClassifier(ClassifierModel):
    def get_probs(self):
        return tf.nn.sigmoid(self.predictions)

    def calculate_loss(self, encoded_text_list, logits_list, labels, l2_weight,
                       dropout_weight=0.0, no_drop_output_list=None, label_smoothing=.0):
        loss = 0.0
        labels = tf.one_hot(labels, self._num_labels, on_value=1.0-label_smoothing, off_value=label_smoothing)
        for logits in logits_list:
            print('----- SIGMOID ----')
            loss_batch = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            ## TODO : example loss weight
            #  loss += tf.reduce_sum(self.example_loss_weights * loss_batch)
            loss += tf.reduce_sum(loss_batch)
        # l2 loss on weights
        if l2_weight > 0.0:
            for et in encoded_text_list:
                loss += tf.constant(l2_weight) * \
                       (tf.nn.l2_loss(self.w_1) + tf.nn.l2_loss(self.w_0)
                        + tf.nn.l2_loss(et)
                        )
        # difference between output with and without dropout
        for et, no_drop_et in zip(encoded_text_list, no_drop_output_list):
            loss += dropout_weight * tf.nn.l2_loss(no_drop_et-et)
        return self.loss_weight * (loss/float(len(logits_list)))


class EntityBinary(ClassifierModel):

    def add_bias(self, predictions):
        self.ep_embeddings = tf.get_variable(name='ep_embeddings',
                                             shape=[2, self._num_labels],
                                             initializer=tf.contrib.layers.xavier_initializer(),)
        ep_embeddings = tf.nn.embedding_lookup(self.ep_embeddings, self.ep_batch)
        predictions += ep_embeddings
        return predictions


    def score_sentence(self, encoded_text, reuse=False, use_seq_len=False):
        with tf.variable_scope('score_sentence', reuse=reuse):
            self.ep_embeddings = tf.get_variable(name='ep_embeddings',
                                                 shape=[2, 1],
                                                 initializer=tf.contrib.layers.xavier_initializer(),)


            ep_embeddings = tf.nn.dropout(tf.nn.embedding_lookup(self.ep_embeddings, self.ep_batch),
                                          self.final_dropout_keep)

            if use_seq_len:
                self.seq_len_embeddings = tf.get_variable(name='seq_len_embeddings',
                                                          shape=[10000, 25],
                                                          initializer=tf.contrib.layers.xavier_initializer())
                seq_embeddings = tf.nn.embedding_lookup(self.seq_len_embeddings, self.seq_len_batch)
                feature_dim = 26
                features = tf.concat(axis=1, values=[ep_embeddings, seq_embeddings, encoded_text])
            else:
                feature_dim = 1
                features = tf.concat(axis=1, values=[ep_embeddings, encoded_text])

            if self.mlp:
                # MLP for scoring encoded sentence
                self.w_1 = tf.get_variable(name='w_1', shape=[self.inner_embed_dim, self._num_labels],
                                           initializer=tf.contrib.layers.xavier_initializer())
                self.b_1 = tf.get_variable(name='b_1', initializer=tf.constant(0.0001, shape=[self._num_labels]))
                self.w_0 = tf.get_variable(name='w_0', shape=[feature_dim+self._embed_dim, self.inner_embed_dim],
                                           initializer=tf.contrib.layers.xavier_initializer())
                self.b_0 = tf.get_variable(name='b_0', initializer=tf.constant(0.0001, shape=[self.inner_embed_dim]))

                prediction_0 = tf.nn.relu(tf.nn.xw_plus_b(features, self.w_0, self.b_0))
                logits = tf.nn.xw_plus_b(prediction_0, self.w_1, self.b_1)
            else:
                # MLP for scoring encoded sentence
                self.w_1 = tf.get_variable(name='w_1', shape=[feature_dim+self.inner_embed_dim, self._num_labels],
                                           # self.w_1 = tf.get_variable(name='w_1', shape=[1+self.inner_embed_dim, self._num_labels],
                                           initializer=tf.contrib.layers.xavier_initializer())
                self.b_1 = tf.get_variable(name='b_1', initializer=tf.constant(0.0001, shape=[self._num_labels]))
                logits = tf.nn.xw_plus_b(features, self.w_1, self.b_1)
            return logits

