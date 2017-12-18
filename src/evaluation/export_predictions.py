from termcolor import colored
import time
from src.feed_dicts import *
import sys

def export_predictions(sess, model, FLAGS, positive_test_batcher, negative_test_batcher,
                       string_int_maps, out_file, threshold_map=None):
    '''
    export hard decisions based on thresholds
    '''
    print('Evaluating')
    null_label_set = set([int(l) for l in FLAGS.null_label.split(',')])
    start_time = time.time()
    pos_eval_epoch = positive_test_batcher.epoch
    neg_eval_epoch = negative_test_batcher.epoch
    i = 0
    scores = []
    pos_count, neg_count = 0, 0
    result_list = [model.probs, model.label_batch, model.e1_batch, model.e2_batch]

    with open(out_file, 'w') as f:
        while positive_test_batcher.epoch == pos_eval_epoch:
            i += 1
            feed_dict, batch_size, doc_ids = batch_feed_dict(positive_test_batcher, sess, model, FLAGS,
                                                             evaluate=True, string_int_maps=string_int_maps)
            # positive batch
            probs, labels, e1, e2 = sess.run(result_list, feed_dict=feed_dict)
            labeled_scores = [(l, np.argmax(s), np.max(s), _e1, _e2, did)
                              for s, l, _e1, _e2, did in zip(probs, labels, e1, e2, doc_ids)]
            scores.append(labeled_scores)
        while negative_test_batcher.epoch == neg_eval_epoch:
            # negative batch
            feed_dict, batch_size, doc_ids = batch_feed_dict(negative_test_batcher, sess, model, FLAGS,
                                                             evaluate=True, string_int_maps=string_int_maps)
            probs, labels, e1, e2 = sess.run(result_list, feed_dict=feed_dict)
            labeled_scores = [(l, np.argmax(s), np.max(s), _e1, _e2, did)
                              for s, l, _e1, _e2, did in zip(probs, labels, e1, e2, doc_ids)]
            neg_count += len(labeled_scores)
            scores.append(labeled_scores)

        # flatten all the batches to a single list
        flat_scores = [x for sublist in scores for x in sublist]
        for label_id in range(FLAGS.num_classes):
            if label_id not in null_label_set:
                label_str = string_int_maps['kb_id_str_map'][label_id]
                thresholds = threshold_map[label_id] if threshold_map \
                    else [float(t) for t in FLAGS.thresholds.split(',')]
                for threshold in thresholds:
                    # label, prediction, confidence
                    predictions = [(_e1, _e2, did) for label, pred, conf, _e1, _e2, did in flat_scores
                                   if conf >= threshold and pred == label_id]
                    mapped_predictions = [(string_int_maps['entity_id_str_map'][_e1],
                                           string_int_maps['entity_id_str_map'][_e2], did)
                                          for _e1, _e2, did in predictions]
                    out_lines = ['%s\t%s\tArg1:%s\tArg2:%s\n' % (did, label_str, _e1, _e2)
                                 for _e1, _e2, did in mapped_predictions]
                    for line in out_lines:
                        f.write(line)

    print('Evaluation took %5.5f seconds' % (time.time() - start_time))
    print('Wrote results to %s' % out_file)


def export_scores(sess, model, FLAGS, positive_test_batcher, negative_test_batcher, string_int_maps, out_file):
    '''
    export all predictions with scores for each label
    '''
    print('Evaluating')
    start_time = time.time()
    pos_eval_epoch = positive_test_batcher.epoch
    neg_eval_epoch = negative_test_batcher.epoch
    i = 0
    result_list = [model.probs, model.label_batch, model.e1_batch, model.e2_batch]

    with open(out_file, 'w') as f:
        while positive_test_batcher.epoch == pos_eval_epoch:
            i += 1
            feed_dict, batch_size, doc_ids = batch_feed_dict(positive_test_batcher, sess, model, FLAGS,
                                                             evaluate=True, string_int_maps=string_int_maps)
            # positive batch
            probs, labels, e1, e2 = sess.run(result_list, feed_dict=feed_dict)
            for p, l, _e1, _e2, did in zip(probs, labels, e1, e2, doc_ids):
                scores = ':'.join([str(_p) for _p in p])
                f.write('%s\tArg1:%s\tArg2:%s\t%s\n'
                        % (did, string_int_maps['entity_id_str_map'][_e1],
                           string_int_maps['entity_id_str_map'][_e2], scores))
        while negative_test_batcher.epoch == neg_eval_epoch:
            feed_dict, batch_size, doc_ids = batch_feed_dict(negative_test_batcher, sess, model, FLAGS,
                                                             evaluate=True, string_int_maps=string_int_maps)
            probs, labels, e1, e2 = sess.run(result_list, feed_dict=feed_dict)
            for p, l, _e1, _e2, did in zip(probs, labels, e1, e2, doc_ids):
                scores = ':'.join([str(_p) for _p in p])
                f.write('%s\tArg1:%s\tArg2:%s\t%s\n'
                        % (did, string_int_maps['entity_id_str_map'][_e1],
                           string_int_maps['entity_id_str_map'][_e2], scores))
    print('Evaluation took %5.5f seconds' % (time.time() - start_time))
    print('Wrote results to %s' % out_file)


def export_attentions(sess, model, FLAGS, positive_test_batcher, negative_test_batcher, string_int_maps, out_file):
    print('Exporting attention weights')
    start_time = time.time()
    pos_eval_epoch = positive_test_batcher.epoch
    result_list = [model.probs, model.label_batch, model.e1_dist_batch, model.e2_dist_batch,
                   model.text_batch, model.text_encoder.attention_weights, model.text_encoder.pairwise_scores]

    batch_num = 0
    attention_values = {}
    pair_values = {}
    take = 500
    with open('%s.txt' % out_file, 'w') as out_f:
        while positive_test_batcher.epoch == pos_eval_epoch and batch_num <= take:
            feed_dict, batch_size, doc_ids = batch_feed_dict(positive_test_batcher, sess, model, FLAGS,
                                                             evaluate=True, string_int_maps=string_int_maps)
            # positive batch
            probs, labels, e1, e2, token_ids, attention_weights, pair_scores = sess.run(result_list, feed_dict=feed_dict)
            pair_scores = [np.transpose(p, (0, 2, 1, 3)) for p in np.dsplit(pair_scores, FLAGS.num_classes)]

            # iterate over a single example
            for example_num, (ex_token_ids, ex_e1s, ex_e2s) in enumerate(zip(token_ids, e1, e2)):
                # convert token ids to strings
                token_strings = [string_int_maps['token_id_str_map'][t] for t in ex_token_ids]
                # generate the strings to write to files
                out_strs = ['batch%d_example%d_token%d\t%s\t%s\t%s\n'
                            % (batch_num, example_num, token_num, token_str, ex_e1s[token_num], ex_e2s[token_num])
                            for token_num, token_str in enumerate(token_strings)]
                out_f.write(''.join(out_strs))
            for layer_num, values in enumerate(attention_weights):
                attention_values['batch%d_layer%d' % (batch_num, layer_num)] = values
            for layer_num, values in enumerate(pair_scores):
                pair_values['batch%d_layer%d' % (batch_num, layer_num)] = values
            batch_num += 1
    np.savez('%s_attention' % out_file, **attention_values)
    np.savez('%s_pairs' % out_file, **pair_values)

    print('Evaluation took %5.5f seconds' % (time.time() - start_time))
    print('Wrote results to %s' % out_file)


# def color_token(t, e1, e2, aw, mean_aws, std_aws, default_background='on_white', default_text='grey'):
#     high_cutoff = (1.96*std_aws)
#     low_cutoff = (0.75*std_aws)
#     # token_color = 'yellow' if e1 == 1 else 'green' if e2 == 1 else 'grey'
#     token_color = 'grey'
#     # background_color = 'on_red' if aw >= _high else 'on_blue' if aw <= _low else 'on_grey'
#     background_color = 'on_red' if aw >= mean_aws+high_cutoff else 'on_magenta' if aw >= mean_aws+low_cutoff \
#         else 'on_blue' if aw <= mean_aws-high_cutoff else 'on_cyan' if aw <= mean_aws-low_cutoff \
#         else default_background
#     # background_color = 'on_red' if aw >= top_5 else 'on_yellow' if aw >= top_10 else 'on_blue' if aw <= bottom_5 else 'on_grey'
#     attributes = ['bold', 'underline'] if e1 == 1 else ['bold' ] if e2 == 1 else ['dark']
#     return colored(t, token_color, background_color, attrs=attributes)
#
#
# def analyze_errors(predictions, string_int_maps):
#     token_id_str_map = string_int_maps['token_id_str_map']
#     shuffle(predictions)
#     false_negatives = [(l, p, c, e1, e2, t, aw) for l, p, c, e1, e2, t, aw in predictions if l != FLAGS.null_label and l != p]
#     false_positives = [(l, p, c, e1, e2, t, aw) for l, p, c, e1, e2, t, aw in predictions if l == FLAGS.null_label and l != p]
#     true_positives = [(l, p, c, e1, e2, t, aw) for l, p, c, e1, e2, t, aw in predictions if l != FLAGS.null_label and l == p]
#     true_negatives = [(l, p, c, e1, e2, t, aw) for l, p, c, e1, e2, t, aw in predictions if l == FLAGS.null_label and l == p]
#
#     # pred_types = {'False Positive': false_positives, 'False Negative': false_negatives,
#     #               'True Positive': true_positives, 'True Negative': true_negatives}
#     pred_types = {'True Positive': true_positives}
#     for _type, _predictions in pred_types.iteritems():
#         e1s = np.mean([np.sum(e1) for l, p, c, e1, e2, t, aw in _predictions])
#         e2s = np.mean([np.sum(e2) for l, p, c, e1, e2, t, aw in _predictions])
#         confs = np.mean([c for l, p, c, e1, e2, t, aw in _predictions])
#         aws = np.mean([np.max(aw) for l, p, c, e1, e2, t, aw in _predictions])
#         print('%s  e1: %2.2f   e2: %2.2f   confs: %2.2f   aws: %2.2f' % (_type, e1s, e2s, confs, aws))
#
#     for _type, _predictions in pred_types.iteritems():
#         print('************************************')
#         for label, pred, conf, e1, e2, tokens, aws in _predictions[:FLAGS.analyze_errors]:
#             # print(aws)
#             # print(tokens.shape, aws.shape)
#             # sys.exit(1)
#             default_background='on_white'
#             default_text='grey'
#             filter_e1, filter_e2, filter_tokens, filter_aws = zip(*[(e1, e2, token_id_str_map[t], aw)
#                                                                     for e1, e2, t, aw in zip(e1, e2, tokens, aws)
#                                                                     if token_id_str_map[t] != '<PAD>'
#                                                                     ])
#             filter_aws = np.squeeze(filter_aws)*100.0
#             mean_aws = np.mean(filter_aws)
#             std_aws = np.std(filter_aws)
#             print('----- %s  -----confidence: %2.2f  ---- min: %2.4f max: %2.4f  mean: %2.4f  std: %2.4f  ----'
#                   % (_type, conf, np.min(filter_aws), np.max(filter_aws), mean_aws, std_aws))
#
#             if 'pairs' not in FLAGS.text_encoder:
#                 token_str_list = [color_token(t, e1, e2, aw, mean_aws, std_aws,
#                                               default_background=default_background, default_text=default_text)
#                                   for e1, e2, t, aw in zip(filter_e1, filter_e2, filter_tokens, filter_aws)]
#             else:
#                 filter_e1 = [x for x in filter_e1]
#                 filter_e2 = [x for x in filter_e2]
#                 filter_aws = np.array([faws if e1 == 1 else (-1e8*np.ones_like(faws))
#                                        for e1, faws in zip(filter_e1, filter_aws)])
#                 filter_aws = np.array([faws if e2 == 1 else (-1e8*np.ones_like(faws))
#                                        for e2, faws in zip(filter_e2, filter_aws.T)])
#                 max_pair = np.unravel_index(filter_aws.argmax(), filter_aws.shape)
#                 color_score = [1 if i == max_pair[0] else -1 if i == max_pair[1] else 0
#                                for i, (e1, e2) in enumerate(zip(filter_e1, filter_e2))]
#                 token_str_list = [color_token(t, e1, e2, cs, 0, .1,
#                                               default_background=default_background, default_text=default_text)
#                                   for i, (e1, e2, t, aw, cs)
#                                   in enumerate(zip(filter_e1, filter_e2, filter_tokens, filter_aws, color_score))]
#
#             token_str = re.sub(r'(@@ )|(@@ ?$)', '', colored(' ', default_text, default_background).join(token_str_list))
#             print(token_str)
#             print('----------------------------------------------------------------------------------')