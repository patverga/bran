# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import tensorflow as tf
from src.models.text_encoders import *
from src.tf_utils import *


class Transformer(TextEncoder):
    def __init__(self, text_batch, e1_dist_batch, e2_dist_batch, ep_dist, seq_len_batch, lstm_dim, embed_dim, position_dim,
                 token_dim, bidirectional, peephole, max_pool, word_dropout_keep, lstm_dropout_keep,
                 final_dropout_keep, entity_index=100, filterwidth=3, block_repeats=1, layer_str='1:1,5:1,1:1',
                 pos_encode_batch=None, filter_pad=False, string_int_maps=None,
                 e1_batch=None, e2_batch=None, entity_embeddings=None, entity_vocab_size=None, num_labels=2, project_inputs=False):

        super(Transformer, self).__init__(text_batch, e1_dist_batch, e2_dist_batch, seq_len_batch, lstm_dim, embed_dim,
                                        position_dim, token_dim, bidirectional, peephole, max_pool, word_dropout_keep,
                                        lstm_dropout_keep, final_dropout_keep, entity_index,
                                        pos_encode_batch=pos_encode_batch, filter_pad=filter_pad,
                                        string_int_maps=string_int_maps, project_inputs=project_inputs)
        self.encoder_type = 'transformer'
        self.nonlinearity = 'relu'
        self.layer_str = layer_str
        self.divisor = self.embed_dim
        self.res_activation = 0
        self.training = False
        self.batch_norm = False
        self.projection = False
        self.filter_width = filterwidth
        self.hidden_dropout_keep_prob = lstm_dropout_keep
        self.middle_dropout_keep_prob = lstm_dropout_keep
        self.block_repeats = block_repeats
        self.encode_position = True
        self.num_heads = 4
        self.ff_scale = 4
        self.num_labels = num_labels
        self.ep_dist_batch = ep_dist

        if self.encode_position:
            max_pos = 10000
            pos_encode_dim = self.embed_dim if self.project_inputs else self.token_dim+(2*self.position_dim)
            self.pos_encoding = tf.get_variable(name='pos_encoding',
                                                shape=[max_pos, pos_encode_dim],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                )

    def embed_text(self, token_embeddings, position_embeddings, attention_vector,
                   scope_name='text', reuse=False, aggregation='attention', no_dropout=False,
                   return_tokens=False, token_attention=None,):
        selected_col_embeddings = self.get_token_embeddings(token_embeddings, position_embeddings, token_attention)
        if no_dropout:
            middle_dropout_keep_prob, hidden_dropout_keep_prob, final_dropout_keep = 1.0, 1.0, 1.0
        else:
            middle_dropout_keep_prob = self.middle_dropout_keep_prob
            hidden_dropout_keep_prob = self.hidden_dropout_keep_prob
            final_dropout_keep = self.final_dropout_keep

        output = self.embed_text_from_tokens(
                selected_col_embeddings, attention_vector,
                self.e1_dist_batch, self.e2_dist_batch, self.seq_len_batch,
                middle_dropout_keep_prob, hidden_dropout_keep_prob, final_dropout_keep,
                scope_name, reuse, aggregation, return_tokens=return_tokens)
        return output

    def embed_text_from_tokens(self, selected_col_embeddings, attention_vector, e1_dist_batch, e2_dist_batch, seq_lens,
                               middle_dropout_keep_prob, hidden_dropout_keep_prob, final_dropout_keep,
                               scope_name='text', reuse=False, aggregation='piecewise', return_tokens=False):
        batch_size = tf.shape(selected_col_embeddings)[0]
        max_seq_len = tf.shape(selected_col_embeddings)[1]

        output = []
        last_output = selected_col_embeddings
        if not reuse:
            print('___aggregation type:  %s filter %d  block repeats: %d___'
                  % (aggregation, self.filter_width, self.block_repeats))
        for i in range(1):
            block_reuse = (reuse if i == 0 else True)
            encoded_tokens = self.forward(last_output, middle_dropout_keep_prob,
                                          hidden_dropout_keep_prob, batch_size, max_seq_len, block_reuse, i)
            if return_tokens:
                output.append(encoded_tokens)
            else:
                encoded_seq = self.aggregate_tokens(encoded_tokens, batch_size, max_seq_len,
                                                    attention_vector, e1_dist_batch, e2_dist_batch, seq_lens,
                                                    middle_dropout_keep_prob, hidden_dropout_keep_prob, final_dropout_keep,
                                                    scope_name=scope_name, reuse=block_reuse, aggregation=aggregation)
                output.append(encoded_seq)
            last_output = encoded_tokens
        return output


    def forward(self, input_feats, middle_dropout_keep_prob, hidden_dropout_keep_prob, batch_size, max_seq_len,
                reuse, block_num=0):
        initial_in_dim =self.embed_dim if self.project_inputs else self.token_dim+(2*self.position_dim)
        input_feats *= (initial_in_dim ** 0.5)
        self.attention_weights = []
        for i in range(self.block_repeats):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ### Multihead Attention
                in_dim = self.embed_dim if i > 0 else initial_in_dim
                input_feats = self.multihead_attention(queries=input_feats,
                                                  keys=input_feats,
                                                  num_units=in_dim,
                                                  num_heads=self.num_heads,
                                                  dropout_rate=middle_dropout_keep_prob,
                                                  causality=False,
                                                  reuse=reuse)

                ### Feed Forward
                input_feats = self.feedforward(input_feats, num_units=[in_dim*self.ff_scale, in_dim], reuse=reuse)
        return input_feats


    def normalize(self, inputs,
                  epsilon = 1e-8,
                  scope="ln",
                  reuse=None):
            '''Applies layer normalization.

            Args:
              inputs: A tensor with 2 or more dimensions, where the first dimension has
                `batch_size`.
              epsilon: A floating number. A very small number for preventing ZeroDivision Error.
              scope: Optional scope for `variable_scope`.
              reuse: Boolean, whether to reuse the weights of a previous layer
                by the same name.

            Returns:
              A tensor with the same shape and data dtype as `inputs`.
            '''
            with tf.variable_scope(scope, reuse=reuse):
                inputs_shape = inputs.get_shape()
                params_shape = inputs_shape[-1:]

                mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
                beta = tf.Variable(tf.zeros(params_shape))
                gamma = tf.Variable(tf.ones(params_shape))
                normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
                outputs = gamma * normalized + beta

            return outputs


    def multihead_attention(self, queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            causality=False,
                            scope="multihead_attention",
                            reuse=None):
        '''
        June 2017 by kyubyong park.
        kbpark.linguist@gmail.com.
        https://www.github.com/kyubyong/transformer
        '''
        '''Applies multihead attention.

        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]

            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs)*(-1e8)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

            # Activation
            attention_weights = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
            # store the attention weights for analysis
            batch_size = tf.shape(queries)[0]
            seq_len = tf.shape(queries)[1]
            save_attention = tf.reshape(attention_weights, [self.num_heads, batch_size, seq_len, seq_len])
            save_attention = tf.transpose(save_attention, [1, 0, 2, 3])
            self.attention_weights.append(save_attention)

            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            outputs = attention_weights * query_masks # broadcasting. (N, T_q, C)

            # Dropouts
            outputs = tf.nn.dropout(outputs, dropout_rate)

            # Weighted sum
            outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1 ) # (N, T_q, C)

            # Residual connection
            outputs += tf.nn.dropout(queries, dropout_rate)

            # Normalize
            outputs = self.normalize(outputs) # (N, T_q, C)

        return outputs


    def feedforward(self, inputs,
                    num_units=[2048, 512],
                    scope="multihead_attention",
                    reuse=None):

        '''Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            outputs = inputs
            layer_params = self.layer_str.split(',')
            for i, l_params in enumerate(layer_params):
                width, dilation = [int(x) for x in l_params.split(':')]
                dim = num_units[1] if i == (len(layer_params)-1) else num_units[0]

                print('dimension: %d  width: %d  dilation: %d' % (dim, width, dilation))
                params = {"inputs": outputs, "filters": dim, "kernel_size": width,
                          "activation": tf.nn.relu, "use_bias": True, "padding": "same", "dilation_rate": dilation}
                outputs = tf.layers.conv1d(**params)
            # mask padding
            outputs *= tf.expand_dims(tf.cast(tf.not_equal(self.text_batch, self.pad_idx), tf.float32), [2])
            # Residual connection
            inputs += outputs

            # Normalize
            outputs = self.normalize(outputs)

        return outputs

    def broadcast_mult(self, inputs1, inputs2):
        """"""

        inputs1_shape = tf.shape(inputs1)
        inputs_size = inputs1.get_shape().as_list()[-1]
        inputs2_shape = tf.shape(inputs2)
        inputs1 = tf.transpose(inputs1, [0,2,1])
        inputs2 = tf.transpose(inputs2, [0,2,1])
        inputs1 = tf.reshape(inputs1, [-1,inputs1_shape[1],1])
        inputs2 = tf.reshape(inputs2, [-1,1,inputs2_shape[1]])
        inputs = inputs1 * inputs2
        inputs = tf.reshape(inputs, [inputs1_shape[0], inputs1_shape[2],  inputs1_shape[1], inputs2_shape[1]])
        inputs = tf.transpose(inputs, [0,2,3,1])
        inputs.set_shape([tf.Dimension(None)]*3 + [tf.Dimension(inputs_size)])
        return inputs

    def linear(self, inputs, output_size, add_bias=True, n_splits=1, initializer=None, scope=None, moving_params=None):
        """"""
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        output_size *= n_splits

        with tf.variable_scope(scope or 'Linear'):
            # Reformat the input
            total_input_size = 0
            shapes = [a.get_shape().as_list() for a in inputs]
            for shape in shapes:
                total_input_size += shape[-1]
            input_shape = tf.shape(inputs[0])
            output_shape = []
            for i in xrange(len(shapes[0])):
                output_shape.append(input_shape[i])
            output_shape[-1] = output_size
            for i, (input_, shape) in enumerate(zip(inputs, shapes)):
                inputs[i] = tf.reshape(input_, [-1, shape[-1]])
            concatenation = tf.concat(inputs, 1)

            # Get the matrix
            if initializer is None and moving_params is None:
                mat = orthonormal_initializer(total_input_size, output_size//n_splits)
                mat = np.concatenate([mat]*n_splits, axis=1)
                initializer = tf.constant_initializer(mat)
            matrix = tf.get_variable('Weights', [total_input_size, output_size], initializer=initializer)
            if moving_params is not None:
                matrix = moving_params.average(matrix)
            else:
                tf.add_to_collection('Weights', matrix)

            # Get the bias
            if add_bias:
                bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer)
                if moving_params is not None:
                    bias = moving_params.average(bias)
            else:
                bias = 0

            # Do the multiplication
            new = tf.matmul(concatenation, matrix) + bias
            new = tf.reshape(new, output_shape)
            new.set_shape([tf.Dimension(None) for _ in xrange(len(shapes[0])-1)] + [tf.Dimension(output_size)])
            if n_splits > 1:
                return tf.split(len(new.get_shape().as_list())-1, n_splits, new)
            else:
                return new

    def bilinear(self, inputs1, inputs2, output_size, add_bias2=True, add_bias1=True, add_bias=False,
                 initializer=None, scope=None, moving_params=None):
        """"""
        with tf.variable_scope(scope or 'Bilinear'):
            # Reformat the inputs
            ndims = len(inputs1.get_shape().as_list())
            inputs1_shape = tf.shape(inputs1)
            inputs1_bucket_size = inputs1_shape[ndims-2]
            inputs1_size = inputs1.get_shape().as_list()[-1]

            inputs2_shape = tf.shape(inputs2)
            inputs2_bucket_size = inputs2_shape[ndims-2]
            inputs2_size = inputs2.get_shape().as_list()[-1]
            output_shape = []
            batch_size = 1
            for i in xrange(ndims-2):
                batch_size *= inputs1_shape[i]
                output_shape.append(inputs1_shape[i])
            output_shape.append(inputs1_bucket_size)
            output_shape.append(output_size)
            output_shape.append(inputs2_bucket_size)
            inputs1 = tf.reshape(inputs1, [batch_size, inputs1_bucket_size, inputs1_size])
            inputs2 = tf.reshape(inputs2, [batch_size, inputs2_bucket_size, inputs2_size])
            if add_bias1:
                inputs1 = tf.concat([inputs1, tf.ones([batch_size, inputs1_bucket_size, 1])], 2)
            if add_bias2:
                inputs2 = tf.concat([inputs2, tf.ones([batch_size, inputs2_bucket_size, 1])], 2)

            # Get the matrix
            if initializer is None and moving_params is None:
                mat = orthonormal_initializer(inputs1_size+add_bias1, inputs2_size+add_bias2)[:,None,:]
                mat = np.concatenate([mat]*output_size, axis=1)
                initializer = tf.constant_initializer(mat)
            weights = tf.get_variable('Weights', [inputs1_size+add_bias1, output_size, inputs2_size+add_bias2], initializer=initializer)
            if moving_params is not None:
                weights = moving_params.average(weights)
            else:
                tf.add_to_collection('Weights', weights)

            # Do the multiplications
            # (bn x d) (d x rd) -> (bn x rd)
            lin = tf.matmul(tf.reshape(inputs1, [-1, inputs1_size+add_bias1]),
                            tf.reshape(weights, [inputs1_size+add_bias1, -1]))
            # (b x nr x d) (b x n x d)T -> (b x nr x n)
            bilin = tf.matmul(tf.reshape(lin, [batch_size, inputs1_bucket_size*output_size, inputs2_size+add_bias2]),
                                    inputs2, adjoint_b=True)
            # (bn x r x n)
            bilin = tf.reshape(bilin, [-1, output_size, inputs2_bucket_size])
            # (b x n x r x n)
            bilin = tf.reshape(bilin, output_shape)

            # Get the bias
            if add_bias:
                bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer)
                if moving_params is not None:
                    bias = moving_params.average(bias)
                bilin += tf.expand_dims(bias, 1)

            return bilin


    def diagonal_bilinear(self, inputs1, inputs2, output_size, add_bias2=True, add_bias1=True, add_bias=False, initializer=None, scope=None, moving_params=None):
        """"""

        with tf.variable_scope(scope or 'Bilinear'):
            # Reformat the inputs
            ndims = len(inputs1.get_shape().as_list())
            inputs1_shape = tf.shape(inputs1)
            inputs2_shape = tf.shape(inputs2)
            inputs1_bucket_size = inputs1_shape[ndims-2]
            inputs2_bucket_size = inputs2_shape[ndims-2]

            inputs1_size = inputs1.get_shape().as_list()[-1]
            inputs2_size = inputs2.get_shape().as_list()[-1]
            assert inputs1_size == inputs2_size

            output_shape = []
            batch_size = 1
            for i in xrange(ndims-2):
                batch_size *= inputs1_shape[i]
                output_shape.append(inputs1_shape[i])
            output_shape.append(inputs1_bucket_size)
            output_shape.append(output_size)
            output_shape.append(inputs2_bucket_size)
            inputs1 = tf.reshape(inputs1, [batch_size, inputs1_bucket_size, inputs1_size])
            inputs2 = tf.reshape(inputs2, [batch_size, inputs2_bucket_size, inputs2_size])
            inputs1.set_shape([tf.Dimension(None)]*2 + [tf.Dimension(inputs1_size)])
            inputs2.set_shape([tf.Dimension(None)]*2 + [tf.Dimension(inputs2_size)])

            inputs = self.broadcast_mult(inputs1, inputs2)
            with tf.variable_scope('Bilinear'):
                bilin = self.linear(inputs, output_size, add_bias=add_bias, initializer=initializer, scope=scope, moving_params=moving_params)
            with tf.variable_scope('Linear1'):
                lin1 = self.linear(inputs1, output_size, add_bias=False, initializer=initializer, scope=scope, moving_params=moving_params)
                lin1 = tf.expand_dims(lin1, 2)
            with tf.variable_scope('Linear2'):
                lin2 = self.linear(inputs2, output_size, add_bias=False, initializer=initializer, scope=scope, moving_params=moving_params)
                lin2 = tf.expand_dims(lin2, 1)

            bilin = tf.transpose(bilin+lin1+lin2, [0,1,3,2])

            return bilin


    def aggregate_tokens(self, encoded_tokens, batch_size, max_seq_len,
                         attention_vector, e1_dist_batch, e2_dist_batch, seq_lens,
                         middle_dropout_keep_prob, hidden_dropout_keep_prob, final_dropout_keep,
                         scope_name='text', reuse=False, aggregation='attention'):

        reduction = tf.reduce_logsumexp
        # # aggregation='attention'
        with tf.variable_scope(scope_name, reuse=reuse):
            input_feats = encoded_tokens

            e1_mask = tf.cast(tf.expand_dims(tf.equal(self.e1_dist_batch, self.entity_index), 2), tf.float32)
            e2_mask = tf.cast(tf.expand_dims(tf.equal(self.e2_dist_batch, self.entity_index), 2), tf.float32)

            # # b x s x (d*l)
            e1 = tf.layers.dense(tf.layers.dense(input_feats, self.embed_dim, activation=tf.nn.relu), self.embed_dim)
            e2 = tf.layers.dense(tf.layers.dense(input_feats, self.embed_dim, activation=tf.nn.relu), self.embed_dim)

            e1 = tf.nn.dropout(e1, final_dropout_keep)
            e2 = tf.nn.dropout(e2, final_dropout_keep)

            # result = self.diagonal_bilinear(e1, e2, num_labels)
            pairwise_scores = self.bilinear(e1, e2, self.num_labels)
            # self.attention_weights = tf.split(self.bilinear_scores, self.num_labels, 2)[1]
            self.pairwise_scores = tf.nn.softmax(pairwise_scores, dim=2)
            result = tf.transpose(pairwise_scores, [0, 1, 3, 2])
            # mask result
            result += tf.expand_dims(self.ep_dist_batch, 3)
            outputs = reduction(result, [1, 2])
            print(outputs.get_shape())

            return outputs


class CNNAllPairs(Transformer):
    def forward(self, input_feats, middle_dropout_keep_prob, hidden_dropout_keep_prob, batch_size, max_seq_len,
                reuse, block_num=0):
        initial_in_dim = self.embed_dim if self.project_inputs else self.token_dim+(2*self.position_dim)
        input_feats *= (initial_in_dim ** 0.5)
        for i in range(self.block_repeats):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ### Multihead Attention
                in_dim = self.embed_dim if i > 0 else initial_in_dim
                # input_feats = self.multihead_attention(queries=input_feats,
                #                                        keys=input_feats,
                #                                        num_units=in_dim,
                #                                        num_heads=self.num_heads,
                #                                        dropout_rate=middle_dropout_keep_prob,
                #                                        causality=False,
                #                                        reuse=reuse)

                ### Feed Forward
                input_feats = self.feedforward(input_feats, num_units=[in_dim*self.ff_scale, in_dim], reuse=reuse)
                input_feats = tf.nn.dropout(input_feats, middle_dropout_keep_prob)
        return input_feats


class GLUAllPairs(Transformer):
    def forward(self, input_feats, middle_dropout_keep_prob, hidden_dropout_keep_prob, batch_size, max_seq_len,
                reuse, block_num=0):
        initial_in_dim = self.embed_dim if self.project_inputs else self.token_dim+(2*self.position_dim)
        # input_feats *= (initial_in_dim ** 0.5)

        with tf.variable_scope('glu', reuse=reuse):
            inputs = input_feats
            layer_params = self.layer_str.split(',')
            for i, l_params in enumerate(layer_params):
                width, dilation = [int(x) for x in l_params.split(':')]
                dim = self.lstm_dim*2

                print('dimension: %d  width: %d  dilation: %d' % (dim, width, dilation))
                params = {"inputs": inputs, "filters": dim, "kernel_size": width,
                          "activation": None, "use_bias": True, "padding": "same", "dilation_rate": dilation}
                outputs = tf.layers.conv1d(**params)

                # apply gate
                output_parts = tf.split(outputs, 2, axis=2)
                gate = tf.nn.sigmoid(output_parts[0])
                outputs = tf.multiply(output_parts[1], gate)

                # mask padding
                outputs *= tf.expand_dims(tf.cast(tf.not_equal(self.text_batch, self.pad_idx), tf.float32), [2])
                # Residual connection
                inputs += outputs
                inputs = self.normalize(inputs)
                inputs = tf.nn.dropout(inputs, middle_dropout_keep_prob)

        print(inputs.get_shape())
        return inputs
