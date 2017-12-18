import time
import random
import re
import sys
import numpy as np
import gzip
from collections import defaultdict
from src.data_utils import *
from src.tf_utils import *
from src.models.classifier_models import *
from src.evaluation.ner_eval import ner_eval
from src.evaluation.relation_eval import relation_eval
from src.evaluation.export_predictions import *
from src.feed_dicts import *
FLAGS = tf.app.flags.FLAGS

pos_ep_sum, pos_ep_count, neg_ep_sum, neg_ep_count = 0, 1, 0, 1


def train_model(model, pos_dist_supervision_batcher, neg_dist_supervision_batcher,
                positive_train_batcher, negative_train_batcher, ner_batcher, sv, sess, saver, train_op, ner_train_op,
                string_int_maps, positive_test_batcher, negative_test_batcher, ner_test_batcher,
                positive_test_test_batcher, negative_test_test_batcher, tac_eval, fb15k_eval,
                text_prob, text_weight,
                log_every, eval_every, neg_samples, save_path,
                kb_pretrain=0, max_decrease_epochs=5, max_steps=-1, assign_shadow_ops=tf.no_op()):
    step = 0.
    examples = 0.
    losses = [1.0]
    loss_idx = 1
    loss_avg_len = 100
    best_score = 0.0
    decrease_epochs = 0
    last_update = time.time()
    eval_every = max(1, int(eval_every/float(FLAGS.text_batch)))
    ner_losses = [1.0]
    ner_loss_idx = 1
    ner_prob = FLAGS.ner_prob
    ner_decay = 1e-4

    # ner_eval(ner_test_batcher, sess, model, token_str_id_map)
    # sys.exit(1)

    print ('Starting training, eval every: %d' % eval_every)
    while not sv.should_stop() and (max_steps <= 0 or step < max_steps) and (decrease_epochs <= max_decrease_epochs):
        # try:
            if FLAGS.anneal_ner and step > 5000:
                ner_prob = ner_prob * 1/(1 + ner_decay * step)
            do_ner_update = random.uniform(0, 1) <= ner_prob
            # eval / serialize
            if step > 0 and step % eval_every == 0:
                sess.run(assign_shadow_ops)
                if positive_test_batcher:
                    avg_p, flat_scores, threshold_map = relation_eval(sess, model, FLAGS,
                                                                      positive_test_batcher, negative_test_batcher,
                                                                      string_int_maps, message='Dev')
                if ner_test_batcher and FLAGS.ner_prob > 0:
                    ner_f1, ner_p = ner_eval(ner_test_batcher, sess, model, FLAGS, string_int_maps)

                keep_score = avg_p if FLAGS.ner_prob < 1.0 else ner_f1
                if keep_score > best_score:
                    decrease_epochs = 0
                    best_score = keep_score
                    if save_path:
                        saved_path = saver.save(sess, save_path)
                        print("Serialized model: %s" % saved_path)
                    if positive_test_batcher:
                        # if FLAGS.analyze_errors > 0: analyze_errors(flat_scores, string_int_maps)
                        if positive_test_test_batcher and negative_test_test_batcher:
                            print('Evaluating Test Test')
                            avg_p, _, _ = relation_eval(sess, model, FLAGS,
                                                        positive_test_test_batcher, negative_test_test_batcher,
                                                        string_int_maps, message='Test', threshold_map=threshold_map)
                # if model doesnt improve after max_decrease_epochs, stop training
                elif FLAGS.ner_prob == 1.0 or not do_ner_update:
                    decrease_epochs += 1
                    print('\nEval decreased for %d epochs out of %d max epochs. Best: %2.2f\n'
                          % (decrease_epochs, max_decrease_epochs, best_score))

            # ner update always
            if do_ner_update:
                ner_batch = ner_batcher.next_batch(sess)
                feed_dict, ner_batch_size = ner_feed_dict(ner_batch, model, FLAGS, string_int_maps=string_int_maps)
                _, global_step, ner_loss, = sess.run([ner_train_op, model.global_step, model.ner_loss], feed_dict=feed_dict)
                if len(ner_losses) < loss_avg_len:
                    ner_losses.append(np.mean(ner_loss))
                    ner_loss_idx += 1
                else:
                    ner_loss_idx = 0 if ner_loss_idx >= (loss_avg_len - 1) else ner_loss_idx + 1
                    ner_losses[ner_loss_idx] = np.mean(ner_loss)
            # relex update
            else:
                # dist supervision update
                if step < kb_pretrain or random.uniform(0, 1) > text_prob:
                    if FLAGS.pos_prob >= random.uniform(0, 1):
                        feed_dict, batch_size, doc_ids = batch_feed_dict(pos_dist_supervision_batcher, sess, model,
                                                                FLAGS, string_int_maps=string_int_maps)
                    else:
                        feed_dict, batch_size, doc_ids = batch_feed_dict(neg_dist_supervision_batcher, sess, model,
                                                                         FLAGS, string_int_maps=string_int_maps)
                else:
                    # text_update
                    if FLAGS.pos_prob >= random.uniform(0, 1):
                        feed_dict, batch_size, doc_ids = batch_feed_dict(positive_train_batcher, sess, model,
                                                                         FLAGS, string_int_maps=string_int_maps)
                    else:
                        feed_dict, batch_size, doc_ids = batch_feed_dict(negative_train_batcher, sess, model,
                                                                         FLAGS, string_int_maps=string_int_maps)
                    feed_dict[model.loss_weight] = FLAGS.text_weight

                feed_dict[model.noise_weight] = FLAGS.variance_min

                _, global_step, loss, = sess.run([train_op, model.global_step, model.loss], feed_dict=feed_dict)
                examples += batch_size
                loss /= batch_size

                # update loss moving avg
                if len(losses) < loss_avg_len:
                    losses.append(loss)
                    loss_idx += 1
                else:
                    loss_idx = 0 if loss_idx >= (loss_avg_len - 1) else loss_idx + 1
                    losses[loss_idx] = loss

            # log
            if step % log_every == 0:
                steps_per_sec = log_every / (time.time() - last_update)
                examples_per_sec = examples / (time.time() - last_update)
                examples = 0.

                sys.stdout.write('\rstep: %d \t avg loss: %.4f \t ner loss: %.4f'
                                 '\t steps/sec: %.4f \t text examples/sec: %5.2f' %
                                 (step, float(np.mean(losses)), float(np.mean(ner_losses)), steps_per_sec, examples_per_sec))
                sys.stdout.flush()
                last_update = time.time()
            step += 1

    print ('\n Done training')
    if best_score > 0.0: print('Best Score: %2.2f' % best_score)


def main(argv):
    ## TODO gross
    if ('transformer' in FLAGS.text_encoder or 'glu' in FLAGS.text_encoder) and FLAGS.token_dim == 0:
        FLAGS.token_dim = FLAGS.embed_dim-(2*FLAGS.position_dim)
    # print flags:values in alphabetical order
    print ('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))

    if FLAGS.vocab_dir == '':
        print('Error: Must supply input data generated from tsv_to_tfrecords.py')
        sys.exit(1)
    if FLAGS.positive_train == '':
        print('Error: Must supply either positive_train')
        sys.exit(1)

    # read in str <-> int vocab maps
    with open(FLAGS.vocab_dir + '/rel.txt', 'r') as f:
        kb_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        kb_id_str_map = {i: s for s, i in kb_str_id_map.iteritems()}
        kb_vocab_size = FLAGS.kb_vocab_size
    with open(FLAGS.vocab_dir + '/token.txt', 'r') as f:
        token_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        if FLAGS.start_end:
            if '<START>' not in token_str_id_map: token_str_id_map['<START>'] = len(token_str_id_map)
            if '<END>' not in token_str_id_map: token_str_id_map['<END>'] = len(token_str_id_map)
        token_id_str_map = {i: s for s, i in token_str_id_map.iteritems()}
        token_vocab_size = len(token_id_str_map)


    with open(FLAGS.vocab_dir + '/entities.txt', 'r') as f:
        entity_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        entity_id_str_map = {i: s for s, i in entity_str_id_map.iteritems()}
        entity_vocab_size = len(entity_id_str_map)
    with open(FLAGS.vocab_dir + '/ep.txt', 'r') as f:
        ep_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        ep_id_str_map = {i: s for s, i in ep_str_id_map.iteritems()}
        ep_vocab_size = len(ep_id_str_map)

    if FLAGS.ner_train != '':
        with open(FLAGS.vocab_dir + '/ner_labels.txt', 'r') as f:
            ner_label_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
            if FLAGS.start_end:
                if '<START>' not in ner_label_str_id_map: ner_label_str_id_map['<START>'] = len(ner_label_str_id_map)
                if '<END>' not in ner_label_str_id_map: ner_label_str_id_map['<END>'] = len(ner_label_str_id_map)
            ner_label_id_str_map = {i: s for s, i in ner_label_str_id_map.iteritems()}
            ner_label_vocab_size = len(ner_label_id_str_map)
    else:
        ner_label_id_str_map = {}
        ner_label_str_id_map = {}
        ner_label_vocab_size = 1
    position_vocab_size = (2 * FLAGS.max_seq)

    label_weights = None
    if FLAGS.label_weights != '':
        with open(FLAGS.label_weights, 'r') as f:
            lines = [l.strip().split('\t') for l in f]
            label_weights = {kb_str_id_map[k]: float(v) for k, v in lines}

    ep_kg_labels = None
    if FLAGS.kg_label_file != '':
        kg_in_file = gzip.open(FLAGS.kg_label_file, 'rb') if FLAGS.kg_label_file.endswith('gz') else open(FLAGS.kg_label_file, 'r')
        lines = [l.strip().split() for l in kg_in_file.readlines()]
        eps = [('%s::%s' % (l[0], l[1]), l[2]) for l in lines]
        ep_kg_labels = defaultdict(set)
        [ep_kg_labels[ep_str_id_map[_ep]].add(pid) for _ep, pid in eps if _ep in ep_str_id_map]
        print('Ep-Kg label map size %d ' % len(ep_kg_labels))
        kg_in_file.close()

    e1_e2_ep_map = {} #{(entity_str_id_map[ep_str.split('::')[0]], entity_str_id_map[ep_str.split('::')[1]]): ep_id
                      #for ep_id, ep_str in ep_id_str_map.iteritems()}
    ep_e1_e2_map = {} #{ep: e1_e2 for e1_e2, ep in e1_e2_ep_map.iteritems()}

    # get entity <-> type maps for sampling negatives
    entity_type_map, type_entity_map = {}, defaultdict(list)
    if FLAGS.type_file != '':
        with open(FLAGS.type_file, 'r') as f:
            entity_type_map = {entity_str_id_map[l.split('\t')[0]]: l.split('\t')[1].strip().split(',') for l in
                               f.readlines() if l.split('\t')[0] in entity_str_id_map}
            for entity, type_list in entity_type_map.iteritems():
                for t in type_list:
                    type_entity_map[t].append(entity)
            # filter
            type_entity_map = {k: v for k, v in type_entity_map.iteritems() if len(v) > 1}
            valid_types = set([t for t in type_entity_map.iterkeys()])
            entity_type_map = {k: [t for t in v if t in valid_types] for k, v in entity_type_map.iteritems()}
            entity_type_map = {k: v for k, v in entity_type_map.iteritems() if len(v) > 1}

    string_int_maps = {'kb_str_id_map': kb_str_id_map, 'kb_id_str_map': kb_id_str_map,
                        'token_str_id_map': token_str_id_map, 'token_id_str_map': token_id_str_map,
                        'entity_str_id_map': entity_str_id_map, 'entity_id_str_map': entity_id_str_map,
                        'ep_str_id_map': ep_str_id_map, 'ep_id_str_map': ep_id_str_map,
                        'ner_label_str_id_map': ner_label_str_id_map, 'ner_label_id_str_map': ner_label_id_str_map,
                        'e1_e2_ep_map': e1_e2_ep_map, 'ep_e1_e2_map': ep_e1_e2_map, 'ep_kg_labels': ep_kg_labels,
                        'label_weights': label_weights}

    word_embedding_matrix = load_pretrained_embeddings(token_str_id_map, FLAGS.embeddings, FLAGS.token_dim, token_vocab_size)
    entity_embedding_matrix = load_pretrained_embeddings(entity_str_id_map, FLAGS.entity_embeddings, FLAGS.embed_dim, entity_vocab_size)

    with tf.Graph().as_default():
        tf.set_random_seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)
        random.seed(FLAGS.random_seed)

        if FLAGS.doc_filter:
            train_percent = FLAGS.train_dev_percent
            with open(FLAGS.doc_filter, 'r') as f:
                doc_filter_ids = [l.strip() for l in f]
            shuffle(doc_filter_ids)
            split_idx = int(len(doc_filter_ids) * train_percent)
            dev_ids, train_ids = set(doc_filter_ids[:split_idx]), set(doc_filter_ids[split_idx:])
            # ids in dev_ids will be filtered from dev, same for train_ids
            print('Splitting dev data %d documents for train and %d documents for dev' % (len(dev_ids), len(train_ids)))
        else:
            dev_ids, train_ids = None, None

        # have seperate batchers for positive and negative train/test
        batcher = InMemoryBatcher if FLAGS.in_memory else Batcher
        pos_dist_supervision_batcher = batcher(FLAGS.positive_dist_train, FLAGS.kb_epochs, FLAGS.max_seq, FLAGS.kb_batch) \
            if FLAGS.positive_dist_train else None
        neg_dist_supervision_batcher = batcher(FLAGS.negative_dist_train, FLAGS.kb_epochs, FLAGS.max_seq, FLAGS.kb_batch) \
            if FLAGS.negative_dist_train else None

        positive_train_batcher = batcher(FLAGS.positive_train, FLAGS.text_epochs, FLAGS.max_seq, FLAGS.text_batch)
        negative_train_batcher = batcher(FLAGS.negative_train, FLAGS.text_epochs, FLAGS.max_seq, FLAGS.text_batch)

        positive_test_batcher = InMemoryBatcher(FLAGS.positive_test, 1, FLAGS.max_seq, FLAGS.text_batch) \
            if FLAGS.positive_test else None
        negative_test_batcher = InMemoryBatcher(FLAGS.negative_test, 1, FLAGS.max_seq, FLAGS.text_batch) \
            if FLAGS.negative_test else None
        positive_test_test_batcher = InMemoryBatcher(FLAGS.positive_test_test, 1, FLAGS.max_seq, FLAGS.text_batch) \
            if FLAGS.positive_test_test else None

        negative_test_test_batcher = InMemoryBatcher(FLAGS.negative_test_test, 1, FLAGS.max_seq, FLAGS.text_batch) \
            if FLAGS.negative_test_test else None
        ner_test_batcher = NERInMemoryBatcher(FLAGS.ner_test, 1, FLAGS.max_seq, 10) if FLAGS.ner_test else None
        ner_batcher = NERBatcher(FLAGS.ner_train, FLAGS.text_epochs, FLAGS.max_seq, FLAGS.ner_batch) \
            if FLAGS.ner_train != '' else None

        # initialize model
        if 'multi' in FLAGS.model_type and 'label' in FLAGS.model_type:
            model_type = MultiLabelClassifier
        elif 'entity' in FLAGS.model_type and 'binary' in FLAGS.model_type:
            model_type = EntityBinary
        else:
            model_type = ClassifierModel
        print('Model type: %s ' % FLAGS.model_type)
        model = model_type(ep_vocab_size, entity_vocab_size, kb_vocab_size, token_vocab_size, position_vocab_size,
                           ner_label_vocab_size, word_embedding_matrix, entity_embedding_matrix, string_int_maps, FLAGS)

        # optimization
        learning_rate = tf.train.exponential_decay(FLAGS.lr, model.global_step, FLAGS.lr_decay_steps,
                                                   FLAGS.lr_decay_rate, staircase=False, name=None)
        print ('Optimizer: %s' % FLAGS.optimizer)
        if FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon, beta2=FLAGS.beta2)
        elif FLAGS.optimizer == 'lazyadam':
            optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon, beta2=FLAGS.beta2)
        elif FLAGS.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif FLAGS.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9, )
        else:
            print('%s is not a supported optimizer type' % FLAGS.optimizer)
            sys.exit(1)

        tvars = tf.trainable_variables()
        if FLAGS.clip_norm > 0:
            if FLAGS.freeze_noise:
                noise_vars = [k for k in tvars if 'noise_classifier' not in k.name]
                if not noise_vars:
                    print('Filtering noise variables removed full graph. Is this the wrong FLAGS.model_type?')
                    sys.exit(1)
                for k in noise_vars:
                    print(k.name)
                grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, noise_vars), FLAGS.clip_norm)
                train_op = optimizer.apply_gradients(zip(grads, noise_vars), global_step=model.global_step)
            else:
                grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), FLAGS.clip_norm)
                if FLAGS.noise_std > 0.0:
                    print('Adding noise to gradients with mean: 0.0 std: %0.3f' % FLAGS.noise_std)
                    noisy_gradients = []
                    for gradient in grads:
                        if gradient is None:
                            noisy_gradients.append(None)
                            continue
                        if isinstance(gradient, tf.IndexedSlices):
                            gradient_shape = gradient.dense_shape
                        else:
                            gradient_shape = gradient.get_shape()
                        std = FLAGS.noise_std
                        scale = tf.sqrt(tf.cast(1+model.global_step, tf.float32))
                        noise = tf.truncated_normal(gradient_shape, stddev=std) / scale
                        noisy_gradients.append(gradient + noise)
                    grads = noisy_gradients
                train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_step)

        else:
            train_op = optimizer.minimize(model.loss, global_step=model.global_step)

        emma = tf.train.ExponentialMovingAverage(decay=0.9999, num_updates=model.global_step)
        emma_op = emma.apply(tvars)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(emma_op)
        ner_train_op = None
        if FLAGS.ner_train != '' and FLAGS.ner_prob > 0:
            ner_grads, _ = tf.clip_by_global_norm(tf.gradients(model.ner_loss, tvars), FLAGS.clip_norm)
            ner_train_op = optimizer.apply_gradients(zip(ner_grads, tvars), global_step=model.global_step)
            with tf.control_dependencies([ner_train_op]):
                ner_train_op = tf.group(emma_op)

        assign_shadow_ops = []
        for t in tvars:
            v_scope_name = t.name.split(":")[0]
            v_scope = '/'.join(v_scope_name.split("/")[:-1])
            v_name = v_scope_name.split("/")[-1]
            try:
                with tf.variable_scope(v_scope, reuse=True):
                    v = tf.get_variable(v_name, t.shape)
                    shadow_v = emma.average(v)
                    assign_shadow_ops.append(v.assign(shadow_v))
            except:
                print('Couldnt get %s' % v_scope_name)

        # restore only variables that exist in the checkpoint - needed to pre-train big models with small models
        if FLAGS.load_model != '':
            reader = tf.train.NewCheckpointReader(FLAGS.load_model)
            cp_list = set([key for key in reader.get_variable_to_shape_map()])
            # if variable does not exist in checkpoint or sizes do not match, dont load
            r_vars = [k for k in tf.global_variables() if k.name.split(':')[0] in cp_list
                        and k.get_shape() == reader.get_variable_to_shape_map()[k.name.split(':')[0]]]
            if len(cp_list) != len(r_vars):
                print('[Warning]: not all variables loaded from file')
                # print('\n'.join(sorted(set(cp_list)-set(r_vars))))
            saver = tf.train.Saver(var_list=r_vars)
        else:
            saver = tf.train.Saver()
        sv = tf.train.Supervisor(logdir=FLAGS.logdir if FLAGS.save_model != '' else None,
                                 global_step=model.global_step,
                                 saver=None,
                                 save_summaries_secs=0,
                                 save_model_secs=0, )

        with sv.managed_session(FLAGS.master,
                                config=tf.ConfigProto(
                                    # log_device_placement=True,
                                    allow_soft_placement=True
                                )) as sess:
            if FLAGS.load_model != '':
                print("Deserializing model: %s" % FLAGS.load_model)
                saver.restore(sess, FLAGS.load_model)

            threads = tf.train.start_queue_runners(sess=sess)
            fb15k_eval = None
            tac_eval = None

            if FLAGS.in_memory:
                if positive_train_batcher: positive_train_batcher.load_all_data(sess, doc_filter=train_ids)
                if negative_train_batcher: negative_train_batcher.load_all_data(sess, doc_filter=train_ids)
            if positive_test_batcher: positive_test_batcher.load_all_data(sess, doc_filter=dev_ids)
            if negative_test_batcher: negative_test_batcher.load_all_data(sess, doc_filter=dev_ids)
            if positive_test_test_batcher: positive_test_test_batcher.load_all_data(sess)
            if negative_test_test_batcher: negative_test_test_batcher.load_all_data(sess)
            if ner_test_batcher: ner_test_batcher.load_all_data(sess)#, test_batches)
            if FLAGS.mode == 'train':
                save_path = '%s/%s' % (FLAGS.logdir, FLAGS.save_model) if FLAGS.save_model != '' else None
                train_model(model, pos_dist_supervision_batcher, neg_dist_supervision_batcher,
                            positive_train_batcher, negative_train_batcher, ner_batcher, sv, sess, saver,
                            train_op, ner_train_op, string_int_maps,
                            positive_test_batcher, negative_test_batcher, ner_test_batcher,
                            positive_test_test_batcher, negative_test_test_batcher,
                            tac_eval, fb15k_eval,
                            FLAGS.text_prob, FLAGS.text_weight,
                            FLAGS.log_every, FLAGS.eval_every, FLAGS.neg_samples,
                            save_path, FLAGS.kb_pretrain, FLAGS.max_decrease_epochs,
                            FLAGS.max_steps, assign_shadow_ops)

            elif FLAGS.mode == 'evaluate':
                print('Evaluating')
                results, _, threshold_map = relation_eval(sess, model, FLAGS, positive_test_batcher,
                                                          negative_test_batcher, string_int_maps)
                if positive_test_test_batcher and negative_test_test_batcher:
                    if FLAGS.export_file != '':
                        export_predictions(sess, model, FLAGS, positive_test_test_batcher, negative_test_test_batcher,
                                           string_int_maps, FLAGS.export_file, threshold_map=threshold_map)
                    else:
                        results, _, _ = relation_eval(sess, model, FLAGS, positive_test_test_batcher,
                                                      negative_test_test_batcher, string_int_maps, threshold_map=threshold_map)
                if ner_test_batcher:
                    ner_eval(ner_test_batcher, sess, model, FLAGS, string_int_maps)
                print (results)
            elif FLAGS.mode == 'export' and FLAGS.export_file != '':
                print('Exporting predictions')
                export_scores(sess, model, FLAGS, positive_test_test_batcher, negative_test_test_batcher,
                              string_int_maps, FLAGS.export_file)
            elif FLAGS.mode == 'attention' and FLAGS.export_file != '':
                print('Exporting attention weights')
                export_attentions(sess, model, FLAGS, positive_test_batcher, negative_test_batcher,
                              string_int_maps, FLAGS.export_file)
            else:
                print('Error: "%s" is not a valid mode' % FLAGS.mode)
                sys.exit(1)

            sv.coord.request_stop()
            sv.coord.join(threads)
            sess.close()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('vocab_dir', '', 'tsv file containing string data')
    tf.app.flags.DEFINE_string('kb_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('positive_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('negative_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('positive_dist_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('negative_dist_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('positive_test', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('negative_test', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('positive_test_test', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('negative_test_test', '',
                                'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('ner_train', '',
                               'file pattern of proto buffers generated from ../src/processing/ner_to_tfrecords.py')
    tf.app.flags.DEFINE_string('ner_test', '',
                               'file pattern of proto buffers generated from ../src/processing/ner_to_tfrecords.py')
    tf.app.flags.DEFINE_string('embeddings', '', 'pretrained word embeddings')
    tf.app.flags.DEFINE_string('entity_embeddings', '', 'pretrained entity embeddings')
    tf.app.flags.DEFINE_string('fb15k_dir', '', 'directory containing fb15k tsv files')
    tf.app.flags.DEFINE_string('nci_dir', '', 'directory containing nci tsv files')
    tf.app.flags.DEFINE_string('noise_dir', '',
                               'directory containing fb15k noise files generated from src/util/generate_noise.py')
    tf.app.flags.DEFINE_string('candidate_file', '', 'candidate file for tac evaluation')
    tf.app.flags.DEFINE_string('variance_file', '', 'variance file in candidate file format')
    tf.app.flags.DEFINE_string('type_file', '', 'tsv mapping entities to types')
    tf.app.flags.DEFINE_string('kg_label_file', '', '13 col tsv for mapping eps -> kg relations')
    tf.app.flags.DEFINE_string('label_weights', '', 'weight examples for unbalanced labels')
    tf.app.flags.DEFINE_string('logdir', '', 'save logs and models to this dir')
    tf.app.flags.DEFINE_string('load_model', '', 'path to saved model to load')
    tf.app.flags.DEFINE_string('save_model', '', 'name of file to serialize model to')
    tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer to use')
    tf.app.flags.DEFINE_string('loss_type', 'softmax', 'optimizer to use')
    tf.app.flags.DEFINE_string('model_type', 'd', 'optimizer to use')
    tf.app.flags.DEFINE_string('text_encoder', 'lstm', 'optimizer to use')
    # todo: make compatitble with cnn and transformer as width:dilation:take
    tf.app.flags.DEFINE_string('layer_str', '1:1,5:1,1:1', 'transformer feed-forward layers (width:dilation)')
    # tf.app.flags.DEFINE_string('layer_str', '1:false,2:false,1:true', 'cnn layers (dilation:take)')
    tf.app.flags.DEFINE_string('variance_type', 'divide', 'type of variance model to use')
    tf.app.flags.DEFINE_string('mode', 'train', 'train, evaluate, analyze')
    tf.app.flags.DEFINE_string('master', '', 'use for Supervisor')
    tf.app.flags.DEFINE_string('doc_filter', '', 'file to dev doc ids to split between train and test')
    tf.app.flags.DEFINE_string('thresholds', '.5,.6,.7,.75,.8,.85,.9,.95,.975', 'thresholds for prediction')
    tf.app.flags.DEFINE_string('null_label', '0', 'index of negative label')
    tf.app.flags.DEFINE_string('export_file', '', 'export predictions to this file in biocreative VI format')

    tf.app.flags.DEFINE_boolean('norm_entities', False, 'normalize entitiy vectors to have unit norm')
    tf.app.flags.DEFINE_boolean('bidirectional', False, 'bidirectional lstm')
    tf.app.flags.DEFINE_boolean('use_tanh', False, 'use tanh')
    tf.app.flags.DEFINE_boolean('use_peephole', False, 'use peephole connections in lstm')
    tf.app.flags.DEFINE_boolean('max_pool', False, 'max pool hidden states of lstm, else take last')
    tf.app.flags.DEFINE_boolean('in_memory', False, 'load data in memory')
    tf.app.flags.DEFINE_boolean('reset_variance', False, 'reset loaded variance projection matrices')
    tf.app.flags.DEFINE_boolean('percentile', False, 'variance weight based off of percentile')
    tf.app.flags.DEFINE_boolean('semi_hard', False, 'use semi hard negative sample selection')
    tf.app.flags.DEFINE_boolean('verbose', False, 'additional logging')
    tf.app.flags.DEFINE_boolean('freeze', False, 'freeze row and column params')
    tf.app.flags.DEFINE_boolean('freeze_noise', False, 'freeze row and column params')
    tf.app.flags.DEFINE_boolean('mlp', False, 'mlp instead of linear for classification')
    tf.app.flags.DEFINE_boolean('debug', False, 'flags for testing')
    tf.app.flags.DEFINE_boolean('start_end', False, 'add start and end tokens to examples')
    tf.app.flags.DEFINE_boolean('filter_pad', False, 'zero out pad token embeddings and attention')
    tf.app.flags.DEFINE_boolean('anneal_ner', False, 'anneal ner prob as training goes on')
    tf.app.flags.DEFINE_boolean('tune_macro_f', False, 'early stopping based on macro F, else micro F')

    # tac eval
    tf.app.flags.DEFINE_boolean('center_only', False, 'only take center in tac eval')
    tf.app.flags.DEFINE_boolean('arg_entities', False, 'replaced entities with arg wildcards')
    tf.app.flags.DEFINE_boolean('norm_digits', True, 'norm digits in tac eval')

    tf.app.flags.DEFINE_float('lr', .01, 'learning rate')
    tf.app.flags.DEFINE_float('epsilon', 1e-8, 'epsilon for adam optimizer')
    tf.app.flags.DEFINE_float('beta2', 0.999, 'beta2 adam optimizer')
    tf.app.flags.DEFINE_float('lr_decay_steps', 25000, 'anneal learning rate every k steps')
    tf.app.flags.DEFINE_float('lr_decay_rate', .75, 'anneal learning rate every k steps')
    tf.app.flags.DEFINE_float('margin', 1.0, 'margin for hinge loss')
    tf.app.flags.DEFINE_float('l2_weight', 0.0, 'weight for l2 loss')
    tf.app.flags.DEFINE_float('dropout_loss_weight', 0.0, 'weight for dropout loss')
    tf.app.flags.DEFINE_float('clip_norm', 1, 'clip gradients to have norm <= this')
    tf.app.flags.DEFINE_float('text_weight', 1.0, 'weight for text updates')
    tf.app.flags.DEFINE_float('ner_weight', 1.0, 'weight for text updates')
    tf.app.flags.DEFINE_float('ner_prob', 0.0, 'probability of drawing a text batch vs kb batch')
    tf.app.flags.DEFINE_float('text_prob', .5, 'probability of drawing a text batch vs kb batch')
    tf.app.flags.DEFINE_float('pos_prob', .5, 'probability of drawing a positive example')
    tf.app.flags.DEFINE_float('f_beta', 1, 'evaluate using the f_beta metric')
    tf.app.flags.DEFINE_float('noise_std', 0, 'add noise to gradients from 0 mean gaussian with this std')
    tf.app.flags.DEFINE_float('train_dev_percent', .6, 'use this portion of dev as additional train data')

    tf.app.flags.DEFINE_float('variance_min', 1.0, 'weight of variance penalty')
    tf.app.flags.DEFINE_float('variance_max', 99.9, 'weight of variance penalty')
    tf.app.flags.DEFINE_float('variance_delta', 0.0, 'increase variance weight by this value each step')
    tf.app.flags.DEFINE_float('pos_noise', 0.0, 'increase variance weight by this value each step')
    tf.app.flags.DEFINE_float('neg_noise', 0.0, 'increase variance weight by this value each step')

    tf.app.flags.DEFINE_float('word_dropout', .9, 'dropout keep probability for word embeddings')
    tf.app.flags.DEFINE_float('word_unk_dropout', 1.0, 'dropout keep probability for word embeddings')
    tf.app.flags.DEFINE_float('pos_unk_dropout', 1.0, 'dropout keep probability for position embeddings')
    tf.app.flags.DEFINE_float('lstm_dropout', 1.0, 'dropout keep probability for lstm output before projection')
    tf.app.flags.DEFINE_float('final_dropout', 1.0, 'dropout keep probability for final row and column representations')

    tf.app.flags.DEFINE_integer('pattern_dropout', 10, 'take this many mentions for rowless')
    tf.app.flags.DEFINE_integer('pos_count', 2206761, 'number of positive training examples')
    tf.app.flags.DEFINE_integer('neg_count', 20252779, 'number of negative training examples')
    tf.app.flags.DEFINE_integer('kb_vocab_size', 237, 'learning rate')
    tf.app.flags.DEFINE_integer('text_batch', 32, 'batch size')
    tf.app.flags.DEFINE_integer('eval_batch', 32, 'batch size')
    tf.app.flags.DEFINE_integer('kb_batch', 4096, 'batch size')
    tf.app.flags.DEFINE_integer('ner_batch', 128, 'batch size')
    tf.app.flags.DEFINE_integer('token_dim', 250, 'token dimension')
    tf.app.flags.DEFINE_integer('lstm_dim', 2048, 'lstm internal dimension')
    tf.app.flags.DEFINE_integer('embed_dim', 100, 'row/col embedding dimension')
    tf.app.flags.DEFINE_integer('position_dim', 5, 'position relative to entities in lstm embedding')
    tf.app.flags.DEFINE_integer('text_epochs', 100, 'train for this many text epochs')
    tf.app.flags.DEFINE_integer('kb_epochs', 100, 'train for this many kb epochs')
    tf.app.flags.DEFINE_integer('kb_pretrain', 0, 'pretrain kb examples for this many steps')
    tf.app.flags.DEFINE_integer('block_repeats', 1, 'apply iterated blocks this many times')
    tf.app.flags.DEFINE_integer('alternate_var_train', 0,
                                'alternate between variance and rest optimizers every k steps')
    tf.app.flags.DEFINE_integer('log_every', 10, 'log every k steps')
    tf.app.flags.DEFINE_integer('eval_every', 10000, 'eval every k steps')
    tf.app.flags.DEFINE_integer('max_steps', -1, 'stop training after this many total steps')
    tf.app.flags.DEFINE_integer('max_seq', 1, 'maximum sequence length')
    tf.app.flags.DEFINE_integer('max_decrease_epochs', 33, 'stop training early if eval doesnt go up')
    tf.app.flags.DEFINE_integer('num_classes', 4, 'number of classes for multiclass classifier')
    tf.app.flags.DEFINE_integer('neg_samples', 200, 'number of negative samples')
    tf.app.flags.DEFINE_integer('random_seed', 1111, 'random seed')
    tf.app.flags.DEFINE_integer('analyze_errors', 0, 'print out error analysis for K examples per type and exit')


tf.app.run()
