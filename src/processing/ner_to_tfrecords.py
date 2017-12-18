from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import tensorflow as tf
import operator
import glob
import gzip
from collections import defaultdict
import multiprocessing
from functools import partial

tf.app.flags.DEFINE_string('in_files', '', 'pattern to match text input files')
tf.app.flags.DEFINE_string('out_dir', '', 'export tf protos')
tf.app.flags.DEFINE_string('load_vocab', '', 'directory containing vocab files to load')
tf.app.flags.DEFINE_integer('num_threads', 12, 'max number of threads to use for parallel processing')
tf.app.flags.DEFINE_integer('min_count', 10, 'max number of threads to use for parallel processing')
tf.app.flags.DEFINE_boolean('padding', 0, '0: no padding, 1: 0 pad to the right of seq, 2: 0 pad to the left')
tf.app.flags.DEFINE_boolean('normalize_digits', False, 'map all digits to 0')
tf.app.flags.DEFINE_boolean('start_end', False, 'add <START> and <END> tokens to sequence')
FLAGS = tf.app.flags.FLAGS

# Helpers for creating Example objects
feature = tf.train.Feature
sequence_example = tf.train.SequenceExample

def features(d): return tf.train.Features(feature=d)
def int64_feature(v): return feature(int64_list=tf.train.Int64List(value=v))
def feature_list(l): return tf.train.FeatureList(feature=l)
def feature_lists(d): return tf.train.FeatureLists(feature_list=d)

queue = multiprocessing.Queue()
queue.put(0)
lock = multiprocessing.Lock()


def update_vocab_counts(line, token_counter, label_map):
    parts = line.strip().split('\t')
    if len(parts) == 4:
        # data format is token \t label \t kg_id \t doc_id'
        token, label, _, _ = parts
        # normalize the digits to all be 0
        token_normalized = re.sub(r'(?<!\$ARG)[0-9]', '0', token) \
            if FLAGS.normalize_digits else token
        token_counter[token_normalized] += 1
        if label not in label_map:
            label_map[label] = len(label_map)
    return 0


def make_example(token_map, label_map, entity_map, token_strs, label_strs, entity_strs, writer):
    if FLAGS.start_end:
        token_strs = ['<START>'] + token_strs + ['<END>']
        entity_strs = ['<UNK>'] + entity_strs + ['<UNK>']
        label_strs = ['<START>'] + label_strs + ['<END>']

    tokens = [token_map[t] if t in token_map else token_map['<UNK>'] for t in token_strs]
    entities = [entity_map[e] if e in entity_map else entity_map['<UNK>'] for e in entity_strs]
    labels = [label_map[l] for l in label_strs]
    seq_len = len(tokens)
    if FLAGS.padding > 0 and len(tokens) < FLAGS.max_len:
        padding = [token_map['<PAD>']] * (FLAGS.max_len - len(tokens))
        tokens = tokens + padding if FLAGS.padding == 1 else padding + tokens
        labels = labels + padding if FLAGS.padding == 1 else padding + labels
        entities = entities + padding if FLAGS.padding == 1 else padding + entities

    tokens = [tf.train.Feature(int64_list=tf.train.Int64List(value=[t])) for t in tokens]
    labels = [tf.train.Feature(int64_list=tf.train.Int64List(value=[l])) for l in labels]
    entities = [tf.train.Feature(int64_list=tf.train.Int64List(value=[e])) for e in entities]
    example = sequence_example(
        context=features({
            'seq_len': int64_feature([seq_len]),
        }),
        feature_lists=feature_lists({
            "tokens": feature_list(tokens),
            "ner_labels": feature_list(labels),
            "entities": feature_list(entities),
        })
    )
    writer.write(example.SerializeToString())
    return 1


def process_file(token_map, label_map, entity_map, total_lines, in_out, log_every=25):
    try:
        in_f, out_path = in_out
        writer = tf.python_io.TFRecordWriter(out_path)
        lines_written = 0
        print('Converting %s to %s' % (in_f, out_path))
        f_reader = gzip.open(in_f, 'rb') if in_f.endswith('.gz') else open(in_f, 'r')
        i = 0
        line = f_reader.readline().strip()
        while line:
            try:
                token_list = []
                label_list = []
                entity_list = []
                i += 1
                # take lines until we reach a blank then create example
                while line:
                    parts = line.strip().split('\t')
                    token_str, label_str, kg_id, _ = parts
                    token_normalized = re.sub(r'(?<!\$ARG)[0-9]', '0', token_str) \
                        if FLAGS.normalize_digits else token_str
                    token_list.append(token_normalized)
                    label_list.append(label_str)
                    entity_list.append(kg_id)
                    line = f_reader.readline().strip()
                    i += 1
                line = f_reader.readline().strip()

                if i % log_every == 0:
                    if not queue.empty():
                        lock.acquire()
                        processed_lines = queue.get(True, .25) + i
                        i = 0
                        queue.put(processed_lines, True, .25)
                        lock.release()
                        if total_lines > 0:
                            percent_done = 100 * processed_lines / float(total_lines)
                            sys.stdout.write('\rProcessing line %d of %d : %2.2f %%'
                                             % (processed_lines, total_lines, percent_done))
                        else:
                            sys.stdout.write('\rProcessing line %d' % processed_lines)
                        sys.stdout.flush()
                lines_written += (make_example(token_map, label_map, entity_map,
                                               token_list, label_list, entity_list, writer))
            except Exception as e:
                print('error', e)
        f_reader.close()

        writer.close()
        print('\nDone processing %s. Wrote %d lines' % (in_f, lines_written))
    except KeyboardInterrupt:
        return 'KeyboardException'


def tsv_to_examples():
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    in_files = sorted(glob.glob(FLAGS.in_files))
    out_files = ['%s/%s.proto' % (FLAGS.out_dir, in_f.split('/')[-1]) for in_f in in_files]

    total_lines = 0
    # iterate over data once to get counts of tokens and entities
    token_counter = defaultdict(int)
    label_map = {'<PAD>': 0, '<START>': 2, '<END>': 1}
    for in_f in in_files:
        if in_f:
            line_num = 0
            errors = 0
            print('Updating vocabs for %s' % in_f)
            f_reader = gzip.open(in_f, 'rb') if in_f.endswith('.gz') else open(in_f, 'r')
            for line in f_reader:
                line_num += 1
                if line_num % 1000 == 0:
                    sys.stdout.write('\rProcessing line: %d \t errors: %d ' % (line_num, errors))
                    sys.stdout.flush()
                errors += update_vocab_counts(line, token_counter, label_map)
            print(' Done')
            f_reader.close()
            total_lines += line_num

    # remove tokens with < min_count
    print('Sorting and filtering vocab maps')
    keep_tokens = sorted([(t, c) for t, c in token_counter.iteritems()
                          if c >= FLAGS.min_count], key=lambda tup: tup[1], reverse=True)
    keep_tokens = [t[0] for t in keep_tokens]

    # export the string->int maps to file
    export_map = [('ner_labels', label_map)]
    # TODO handle generting new entities map
    entity_map = {'<UNK>': 0}
    if FLAGS.load_vocab:
        print('Loading vocab from %s' % FLAGS.load_vocab)
        with open('%s/token.txt' % FLAGS.load_vocab) as f:
            token_map = {l.split('\t')[0]: int(l.split('\t')[1]) for l in f}
        with open('%s/entities.txt' % FLAGS.load_vocab) as f:
            entity_map = {l.split('\t')[0]: int(l.split('\t')[1]) for l in f}
        print('Loaded %d tokens' % (len(token_map)))
    else:
        # int map all the kept vocab strings
        token_map = {t: i for i, t in enumerate(['<PAD>', '<UNK>'] + keep_tokens)}
        export_map.append(('token', token_map))
    for f_str, id_map in export_map:
        print('Exporting vocab maps to %s/%s' % (FLAGS.out_dir, f_str))
        with open('%s/%s.txt' % (FLAGS.out_dir, f_str), 'w') as f:
            sorted_id_map = sorted(id_map.items(), key=operator.itemgetter(1))
            [f.write(s + '\t' + str(i) + '\n') for (s, i) in sorted_id_map]

    print('Starting file process threads using %d threads' % FLAGS.num_threads)
    pool = multiprocessing.Pool(FLAGS.num_threads)
    try:
        pool.map_async(partial(process_file, token_map, label_map, entity_map, total_lines),
                       zip(in_files, out_files)).get(999999)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()


def main(argv):
    print('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
    if FLAGS.out_dir == '':
        print('Must supply out_dir')
        sys.exit(1)
    tsv_to_examples()


if __name__ == '__main__':
    tf.app.run()
