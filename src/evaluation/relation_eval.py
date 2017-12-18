import time
from src.feed_dicts import *


def relation_eval(sess, model, FLAGS, positive_test_batcher, negative_test_batcher,
                  string_int_maps, message="", threshold_map=None):
    print('Evaluating')
    null_label_set = set([int(l) for l in FLAGS.null_label.split(',')])
    null_label = FLAGS.null_label.split(',')[0]
    start_time = time.time()
    pos_eval_epoch = positive_test_batcher.epoch
    neg_eval_epoch = negative_test_batcher.epoch
    neg_upsample = 1
    i = 0
    scores = []

    pos_count, neg_count = 0, 0
    result_list = [model.probs, model.label_batch, model.e1_dist_batch, model.e2_dist_batch,
                   model.text_batch, model.text_encoder.attention_weights] \
        if FLAGS.analyze_errors > 0 else [model.probs, model.label_batch]

    while positive_test_batcher.epoch == pos_eval_epoch:
        i += 1
        feed_dict, batch_size, doc_ids = batch_feed_dict(positive_test_batcher, sess, model, FLAGS,
                                                         evaluate=True, string_int_maps=string_int_maps)
        if i % neg_upsample == 0:
            # positive batch
            output = sess.run(result_list, feed_dict=feed_dict)
            if FLAGS.analyze_errors > 0:
                labeled_scores = [(l, np.argmax(s), np.max(s), e1, e2, tokens, aw) for s, l, e1, e2, tokens, aw in zip(*output)]
            else:
                labeled_scores = [(l, np.argmax(s), np.max(s)) for s, l in zip(*output)]
            # pos_count += len(labeled_scores)
            scores.append(labeled_scores)
    while negative_test_batcher.epoch == neg_eval_epoch:
        # negative batch
        feed_dict, batch_size, doc_ids = batch_feed_dict(negative_test_batcher, sess, model, FLAGS,
                                                         evaluate=True, string_int_maps=string_int_maps)
        output = sess.run(result_list, feed_dict=feed_dict)
        if FLAGS.analyze_errors > 0:
            labeled_scores = [(l, np.argmax(s), np.max(s), e1, e2, tokens, aw) for s, l, e1, e2, tokens, aw in zip(*output)]
        else:
            labeled_scores = [(l, np.argmax(s), np.max(s)) for s, l in zip(*output)]
        neg_count += len(labeled_scores)
        scores.append(labeled_scores)

    # flatten all the batches to a single list
    flat_scores = [x for sublist in scores for x in sublist]
    # throw out things that were predicted null
    labeled_pos_counts = {label_id: len([x for x in flat_scores if x[0] == label_id])
                          for label_id in range(FLAGS.num_classes) if label_id not in null_label_set}
    best_scores = {}
    for label_id in range(FLAGS.num_classes):
        if label_id not in null_label_set:
            pos_count = labeled_pos_counts[label_id]
            best_label_f = 0.0
            best_label_p = 0.0
            best_label_r = 0.0
            best_correct = 0.0
            best_taken = 0.0
            best_label_threshold = .5
            thresholds = threshold_map[label_id] if threshold_map \
                else [float(t) for t in FLAGS.thresholds.split(',')]
            for threshold in thresholds:
                # label, prediction, confidence
                threshold_scores = [(x[0], x[1]) if x[2] >= threshold else (x[0], null_label) for x in flat_scores]
                predictions = [(label, pred) for label, pred in threshold_scores if pred not in null_label_set]
                label_predictions = [(l, p) for l, p in predictions if p == label_id]
                taken = len(label_predictions)
                correct = len([l for l, p in label_predictions if l == label_id and l == p])
                precision = 100*(correct / float(taken)) if taken > 0 else 0.0
                recall = 100*(correct / float(pos_count)) if pos_count > 0 else 0.0
                f_score = tf_utils.calc_f_score(precision, recall, FLAGS.f_beta)
                if f_score > best_label_f:
                    best_label_p = precision
                    best_label_r = recall
                    best_label_f = f_score
                    best_taken = taken
                    best_correct = correct
                    best_label_threshold = threshold
            # keep the best threshold scores for each label to calc macro / micro later
            best_scores[label_id] = (best_label_p, best_label_r, best_label_f, best_correct, best_taken, best_label_threshold)
            print('pos examples: %d   neg examples:  %d  correct : %d  taken : %d'
                  % (pos_count, neg_count, best_correct, best_taken))
            print('precision: %2.2f   recall: %2.2f   f: %2.4f   threshold: %2.4f    label: %s'
                  % (best_label_p, best_label_r, best_label_f, best_label_threshold, string_int_maps['kb_id_str_map'][label_id]))

    # macro F1 for this threshold
    macro_p = np.mean([p for p, r, f, c, take, thresh in best_scores.itervalues()])
    macro_r = np.mean([r for p, r, f, c, take, thresh in best_scores.itervalues()])
    macro_f = tf_utils.calc_f_score(macro_p, macro_r, FLAGS.f_beta)

    # micro F1 for this threshold
    all_correct = np.sum([c for p, r, f, c, take, thresh in best_scores.itervalues()])
    all_taken = np.sum([take for p, r, f, c, take, thresh in best_scores.itervalues()])
    all_positive = np.sum(labeled_pos_counts.values())
    micro_p = 100*(all_correct / float(all_taken)) if all_taken > 0 else 0.0
    micro_r = 100*(all_correct / float(all_positive)) if all_positive > 0 else 0.0
    micro_f = tf_utils.calc_f_score(micro_p, micro_r, FLAGS.f_beta)

    print('precision: %2.2f   recall: %2.2f   f: %2.4f label: %s'
          % (macro_p, macro_r, macro_f, 'Macro F1'))
    print('precision: %2.2f   recall: %2.2f   f: %2.4f label: %s'
          % (micro_p, micro_r, micro_f, 'Micro F1'))
    print('')

    best_f = macro_f if FLAGS.tune_macro_f else micro_f
    print('%s\tBest F: %2.4f' % (message, best_f))
    print('Evaluation took %5.5f seconds' % (time.time() - start_time))
    if FLAGS.analyze_errors == 0:
        flat_scores = []
    best_threshold_map = {label_id: [thresh] for label_id, (p, r, f, c, take, thresh) in best_scores.iteritems()}
    return best_f, flat_scores, best_threshold_map