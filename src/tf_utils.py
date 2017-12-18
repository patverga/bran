from __future__ import division
import tensorflow as tf
import numpy as np

eps = 1e-5


def embedding_nearest_neighbors(embeddings, batch=None):
    normalized_embeddings = tf.contrib.layers.unit_norm(embeddings, 1, .00001)
    selected_embeddings = tf.nn.embedding_lookup(normalized_embeddings, batch) \
        if batch is not None else normalized_embeddings
    similarity = tf.matmul(selected_embeddings, normalized_embeddings, transpose_b=True)
    return similarity


def gather_nd(params, indices, shape=None, name=None):
    if shape is None:
        shape = params.get_shape().as_list()
    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x * y, shape[i + 1:], 1) for i in range(0, rank)]
    indices_unpacked = tf.unstack(tf.cast(tf.transpose(indices, [rank - 1] + range(0, rank - 1), name), 'int32'))
    flat_indices = sum([a * b for a, b in zip(multipliers, indices_unpacked)])
    return tf.gather(flat_params, flat_indices, name=name)


def repeat(tensor, reps):
    flat_tensor = tf.reshape(tensor, [-1, 1])  # Convert to a len(yp) x 1 matrix.
    repeated = tf.tile(flat_tensor, [1, reps])  # Create multiple columns.
    repeated_flat = tf.reshape(repeated, [-1])  # Convert back to a vector.
    return repeated_flat


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def word_dropout(token_batch, keep_prob, pad_id=0, unk_id=1):
    """ apply word dropout"""
    # create word dropout mask
    word_probs = np.random.random(token_batch.shape)
    drop_indices = np.where((word_probs > keep_prob) & (token_batch != pad_id))
    token_batch[drop_indices[0], drop_indices[1]] = unk_id
    return token_batch


def attention(keys, values, query, filter=None, message=None, scaled=True):
    attention_expanded = tf.expand_dims(query, [2])
    scale = tf.sqrt(tf.cast(tf.shape(query)[-1], tf.float32)) if scaled else 1.0
    attention_scores = tf.matmul(keys, attention_expanded) / scale
    if filter is not None:
        attention_scores = tf.add(attention_scores, filter)
    attention_weights = tf.nn.softmax(attention_scores, dim=1)
    # if message is not None:
    #     attention_weights = tf.Print(attention_weights, [tf.shape(attention_weights),
    #                                                      tf.reduce_min(attention_weights[0]),
    #                                                      tf.reduce_max(attention_weights[0]),
    #                                                      tf.reduce_mean(attention_weights[0])],
    #                                  message=message)
    weighted_tokens = tf.multiply(values, attention_weights)
    return weighted_tokens, attention_weights


def apply_nonlinearity(parameters, nonlinearity_type):
    if nonlinearity_type == "relu":
        return tf.nn.relu(parameters, name="relu")
    elif nonlinearity_type == "tanh":
        return tf.nn.tanh(parameters, name="tanh")
    elif nonlinearity_type == "sigmoid":
        return tf.nn.sigmoid(parameters, name="sigmoid")


def initialize_weights(shape, name, init_type, gain="1.0", divisor=1.0):
    if init_type == "random":
        return tf.get_variable(name, initializer=tf.truncated_normal(shape, stddev=0.1))
    if init_type == "xavier":
        # shape_is_tensor = issubclass(type(shape), tf.Tensor)
        # rank = len(shape.get_shape()) if shape_is_tensor else len(shape)
        # if rank == 4:
        #     return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    if init_type == "identity":
        middle = int(shape[1] / 2)
        if shape[2] == shape[3]:
            array = np.zeros(shape, dtype='float32')
            identity = np.eye(shape[2], shape[3])
            array[0, middle] = identity
        else:
            m1 = divisor / shape[2]
            m2 = divisor / shape[3]
            sigma = eps*m2
            array = np.random.normal(loc=0, scale=sigma, size=shape).astype('float32')
            for i in range(shape[2]):
                for j in range(shape[3]):
                    if int(i*m1) == int(j*m2):
                        array[0, middle, i, j] = m2
        return tf.get_variable(name, initializer=array)
    if init_type == "varscale":
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    if init_type == "orthogonal":
        gain = np.sqrt(2) if gain == "relu" else 1.0
        array = np.zeros(shape, dtype='float32')
        random = np.random.normal(0.0, 1.0, (shape[2], shape[3])).astype('float32')
        u, _, v_t = np.linalg.svd(random, full_matrices=False)
        middle = int(shape[1] / 2)
        array[0, middle] = gain * v_t
        return tf.get_variable(name, initializer=array)


def residual_layer(input, w, b, filter_width, dilation, nonlinearity, batch_norm,
                   name, batch_size, max_sequence_len, activation, training):
    # if activation == "pre" (2): BN -> relu -> weight -> BN -> relu -> weight -> addition
    conv_in_bn = tf.contrib.layers.batch_norm(input, decay=0.995, scale=False, is_training=training, trainable=True) \
                    if batch_norm and activation == 2 else input
    conv_in = apply_nonlinearity(conv_in_bn, nonlinearity) if activation == 2 else conv_in_bn

    conv = tf.nn.atrous_conv2d(
        conv_in,
        w,
        rate=dilation,
        padding="SAME",
        name=name) \
        if dilation > 1 else \
        tf.nn.conv2d(conv_in, w, strides=[1, filter_width, 1, 1], padding="SAME", name=name)

    conv_b = tf.nn.bias_add(conv, b)
    # return conv_b

    # if activation == "post" (1): weight -> BN -> relu -> weight -> BN -> addition -> relu
    conv_out_bn = tf.contrib.layers.batch_norm(conv_b, decay=0.995, scale=False, is_training=training, trainable=True) \
                    if batch_norm and activation != 2 else conv_b
    conv_out = apply_nonlinearity(conv_out_bn, nonlinearity) if activation != 2 else conv_out_bn
    # if activation == "none" (0): weight -> BN -> relu
    conv_shape = w.get_shape()
    if conv_shape[-1] != conv_shape[-2] and activation != 0:
        # if len(input_shape) != 2:
        input = tf.reshape(input, [-1, tf.to_int32(conv_shape[-2])])
        w_r = initialize_weights([conv_shape[-2], conv_shape[-1]], "w_o_" + name, init_type="xavier")
        b_r = tf.get_variable("b_r_" + name, initializer=tf.constant(0.01, shape=[conv_shape[-1]]))
        input_projected = tf.nn.xw_plus_b(input, w_r, b_r, name="proj_r_" + name)
        # if len(output_shape) != 2:
        input_projected = tf.reshape(input_projected, tf.stack([batch_size, 1, max_sequence_len, tf.to_int32(conv_shape[-1])]))
        return tf.add(input_projected, conv_out)
    else:
        return conv_out


def orthonormal_initializer(input_size, output_size):
    """"""
    print(tf.get_variable_scope().name)
    I = np.eye(output_size)
    lr = .1
    eps = .05/(output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in xrange(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI**2 / 2)
            Q2 = Q**2
            Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return Q.astype(np.float32)


def calc_f_score(precision, recall, beta=1):
    if precision + recall <= 0:
        return 0.0
    return (1+(beta**2)) * ((precision * recall) /
                            (((beta**2)*precision) + recall))


def load_pretrained_embeddings(str_id_map, embedding_file, dim, vocab_size):
    # load embeddings, if given; initialize in range [-.01, .01]
    preloaded_embeddings = {}
    embeddings_used = 0
    var = 0
    if embedding_file != '':
        print('Loading embeddings from %s ' % embedding_file)
        with open(embedding_file, 'r') as f:
            for line in f.readlines():
                key, value_str = line.strip().split(' ', 1)
                if key not in str_id_map and key+'@' in str_id_map:
                    key += '@'
                if key in str_id_map:
                    preloaded_vector = [float(v) for v in value_str.split(' ')]
                    if len(preloaded_vector) == dim:
                        embeddings_used += 1
                        v = np.array(preloaded_vector)
                        if v.shape[0] == dim:
                            var += np.var(v)
                            preloaded_embeddings[key] = v
    print("Loaded %d/%d embeddings (%2.2f%% coverage)" % (
        embeddings_used, vocab_size, embeddings_used / float(vocab_size) * 100))
    alpha = (var / len(preloaded_embeddings)) if var != 0 else .1
    normalizer = 1000.0
    print('alpha: %2.3f' % alpha)
    embedding_matrix = np.array([preloaded_embeddings[t] / normalizer if t in preloaded_embeddings
                                 else (np.sqrt(6.0 / (np.sum(dim))))
                                      * np.random.uniform(low=-alpha, high=alpha, size=dim)
                                 for t in str_id_map.iterkeys()])
    return embedding_matrix
