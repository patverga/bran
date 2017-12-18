import numpy as np
import src.tf_utils as tf_utils

def batch_feed_dict(batcher, sess, model, FLAGS, evaluate=False, string_int_maps=None):
    batch = batcher.next_batch(sess)
    e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len, doc_ids = batch
    tokens = tokens if evaluate else tf_utils.word_dropout(tokens, FLAGS.word_unk_dropout)
    e1_dist = e1_dist if evaluate else tf_utils.word_dropout(e1_dist, FLAGS.pos_unk_dropout, unk_id=0)
    e2_dist = e2_dist if evaluate else tf_utils.word_dropout(e2_dist, FLAGS.pos_unk_dropout, unk_id=0)

    ep_dist = np.full((e1_dist.shape[0], e1_dist.shape[1], e1_dist.shape[1]), -1e8)
    ep_indices = [(b, ei, ej) for b in range(e1_dist.shape[0])
                  for ei in np.where(e1_dist[b] == 1)[0]
                  for ej in np.where(e2_dist[b] == 1)[0]]
    b, r, c = zip(*ep_indices)
    ep_dist[b, r, c] = 0.0

    if FLAGS.start_end:
        end_id = string_int_maps['token_str_id_map']['<END>']
        start_id = string_int_maps['token_str_id_map']['<START>']
        zero_col = np.zeros((seq_len.shape[0], 1))
        start_col = zero_col + start_id
        tokens = np.hstack((start_col, tokens, zero_col))
        for i, s in enumerate(seq_len):
            tokens[i, s+1] = end_id
        e1_dist = np.hstack((zero_col, e1_dist, zero_col))
        e2_dist = np.hstack((zero_col, e2_dist, zero_col))
        seq_len += 2

    if FLAGS.kg_label_file:
        # dont use label from this document
        ep = np.array([(1 if len(string_int_maps['ep_kg_labels'][_ep]) > 1
                             or _did not in string_int_maps['ep_kg_labels'][_ep]
                        else 0)
                       if _ep in string_int_maps['ep_kg_labels']
                       else 0 for _ep, _did in zip(ep, doc_ids)])

    pos_encode = [range(1, tokens.shape[1]+1) for i in range(tokens.shape[0])]
    # if FLAGS.num_classes == 2:
    #     label_batch = np.ones(e1.shape) if positive else np.zeros(e1.shape)
    # else:
    label_batch = rel
    if string_int_maps['label_weights']:
        lw = string_int_maps['label_weights']
        ex_loss = [lw[l] if l in lw else 1.0 for l in label_batch]
    else:
        ex_loss = np.ones_like(e1)

    feed_dict = {model.text_batch: tokens, model.e1_dist_batch: e1_dist, model.e2_dist_batch: e2_dist,
                 model.seq_len_batch: seq_len, model.label_batch: label_batch, model.pos_encode_batch: pos_encode,
                 model.ep_batch: ep, model.kb_batch: rel, model.e1_batch: e1, model.e2_batch: e2,
                 model.example_loss_weights: ex_loss, model.ep_dist_batch: ep_dist}

    if not evaluate:
        feed_dict[model.word_dropout_keep] = FLAGS.word_dropout
        feed_dict[model.lstm_dropout_keep] = FLAGS.lstm_dropout
        feed_dict[model.final_dropout_keep] = FLAGS.final_dropout

    return feed_dict, e1.shape[0], doc_ids


def ner_feed_dict(batch, model, FLAGS, evaluate=False, string_int_maps=None):
    tokens, labels, entities, seq_len = batch
    tokens = tokens if evaluate else tf_utils.word_dropout(tokens, FLAGS.word_unk_dropout)

    if FLAGS.start_end:
        end_id = string_int_maps['token_str_id_map']['<END>']
        start_id = string_int_maps['token_str_id_map']['<START>']
        ner_start_id = string_int_maps['ner_label_str_id_map']['<START>']
        ner_end_id = string_int_maps['ner_label_str_id_map']['<END>']
        zero_col = np.zeros((seq_len.shape[0], 1))
        start_col = zero_col + start_id
        ner_start_col = zero_col + ner_start_id
        tokens = np.hstack((start_col, tokens, zero_col))
        labels = np.hstack((ner_start_col, labels, zero_col))
        for i, s in enumerate(seq_len):
            tokens[i, s+1] = end_id
            labels[i, s+1] = ner_end_id
        seq_len += 2
    e1_dist = np.zeros_like(tokens)
    e2_dist = np.zeros_like(tokens)
    pos_encode = [range(1, tokens.shape[1]+1) for i in range(tokens.shape[0])]

    feed_dict = {model.text_batch: tokens, model.seq_len_batch: seq_len, model.ner_label_batch: labels,
                 model.e1_dist_batch: e1_dist, model.e2_dist_batch: e2_dist, model.loss_weight: FLAGS.ner_weight,
                 model.pos_encode_batch: pos_encode}
    if not evaluate:
        feed_dict[model.word_dropout_keep] = FLAGS.word_dropout
        feed_dict[model.lstm_dropout_keep] = FLAGS.lstm_dropout
        feed_dict[model.final_dropout_keep] = FLAGS.final_dropout
    return feed_dict, tokens.shape[0]
