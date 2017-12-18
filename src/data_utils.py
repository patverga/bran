import sys
import time
import numpy as np
import tensorflow as tf
from collections import defaultdict
from random import shuffle

FLAGS = tf.app.flags.FLAGS


class Batcher(object):
    def __init__(self, in_file, num_epochs, max_seq, batch_size):
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._max_seq = max_seq
        self.step = 1.
        self.in_file = in_file
        self.next_batch_op = self.input_pipeline(in_file, self._batch_size, num_epochs=num_epochs)

    def next_batch(self, sess):
        return sess.run(self.next_batch_op)

    def input_pipeline(self, file_pattern, batch_size, num_epochs=None, num_threads=10):
        filenames = tf.matching_files(file_pattern)
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        parsed_batch = self.example_parser(filename_queue)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 12 * batch_size
        next_batch = tf.train.batch(
                parsed_batch, batch_size=batch_size, capacity=capacity,
                num_threads=num_threads, dynamic_pad=True, allow_smaller_final_batch=True)
        return next_batch

    def example_parser(self, filename_queue):
        reader = tf.TFRecordReader()
        key, record_string = reader.read(filename_queue)

        # Define how to parse the example
        context_features = {
            'doc_id': tf.FixedLenFeature([], tf.string),
            'e1': tf.FixedLenFeature([], tf.int64),
            'e2': tf.FixedLenFeature([], tf.int64),
            'ep': tf.FixedLenFeature([], tf.int64),
            'rel': tf.FixedLenFeature([], tf.int64),
            'seq_len': tf.FixedLenFeature([], tf.int64),
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "e1_dist": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "e2_dist": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=record_string,
                                                                           context_features=context_features,
                                                                           sequence_features=sequence_features)
        doc_id = context_parsed['doc_id']
        e1 = context_parsed['e1']
        e2 = context_parsed['e2']
        ep = context_parsed['ep']
        rel = context_parsed['rel']
        tokens = sequence_parsed['tokens']
        e1_dist = sequence_parsed['e1_dist']
        e2_dist = sequence_parsed['e2_dist']
        seq_len = context_parsed['seq_len']

        return [e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len, doc_id]


class InMemoryBatcher(Batcher):
    def __init__(self, in_file, num_epochs, max_seq, batch_size):
        super(InMemoryBatcher, self).__init__(in_file, num_epochs, max_seq, batch_size)
        self.epoch = 0.
        loading_batch_size = self._batch_size
        self.next_batch_op = self.input_pipeline(in_file, loading_batch_size, num_epochs=1, num_threads=1)
        self.data = defaultdict(list)
        self._starts = {}
        self._ends = {}
        self._bucket_probs = {}

    def load_all_data(self, sess, max_batches=-1, pad=0, bucket_space=0, doc_filter=None):
        '''
        load batches to memory for shuffling and dynamic padding
        '''
        batch_num = 0
        samples = 0
        start_time = time.time()
        print ('Loading data from %s with batch size: %d' % (self.in_file, self._batch_size))
        try:
            while max_batches <= 0 or batch_num < max_batches:
                batch = sess.run(self.next_batch_op)
                e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len, doc_id = batch

                batch = [(_e1, _e2, _ep, _rel, _tokens, _e1_dist, _e2_dist, _seq_len, _doc_id)
                         for (_e1, _e2, _ep, _rel, _tokens, _e1_dist, _e2_dist, _seq_len, _doc_id)
                         in zip(e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len, doc_id)
                         if not doc_filter or _doc_id not in doc_filter]
                if batch:
                    e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len, doc_id = zip(*batch)
                    e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len, doc_id = \
                        np.array(e1), np.array(e2), np.array(ep), np.array(rel), np.array(tokens), \
                        np.array(e1_dist), np.array(e2_dist), np.array(seq_len), np.array(doc_id)

                    # pad sequences a little bit so buckets aren't so sparse
                    if bucket_space > 0:
                        add_pad = (bucket_space - tokens.shape[1] % bucket_space)
                        zero_col = np.ones((seq_len.shape[0], add_pad)) * pad
                        tokens = np.hstack((tokens, zero_col))
                        e1_dist = np.hstack((e1_dist, zero_col))
                        e2_dist = np.hstack((e2_dist, zero_col))
                    samples += e1.shape[0]
                    updated_batch = (e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len, doc_id)
                    self.data[tokens.shape[1]].append(updated_batch)
                    batch_num += 1
                    sys.stdout.write('\rLoading batch: %d' % batch_num)
                    sys.stdout.flush()
        except Exception as e:
            print('')
        for seq_len, batches in self.data.iteritems():
            self.data[seq_len] = [tuple((e1[i], e2[i], ep[i], rel[i], tokens[i], e1d[i], e2d[i], sl[i], did[i]))
                                  for (e1, e2, ep, rel, tokens, e1d, e2d, sl, did) in batches
                                  for i in range(e1.shape[0])]
        self.reset_batch_pointer()
        end_time = time.time()
        print('Done, loaded %d samples in %5.2f seconds' % (samples, (end_time-start_time)))
        return batch_num

    def next_batch(self, sess):
        # select bucket to create batch from
        self.step += 1
        bucket = self.select_bucket()
        batch = self.data[bucket][self._starts[bucket]:self._ends[bucket]]
        # update pointers
        self._starts[bucket] = self._ends[bucket]
        self._ends[bucket] = min(self._ends[bucket] + self._batch_size, len(self.data[bucket]))
        self._bucket_probs[bucket] = max(0, len(self.data[bucket]) - self._starts[bucket])

        #TODO this is dumb
        _e1 = np.array([e1 for e1, e2, ep, rel, t, e1d, e2d, s, did in batch])
        _e2 = np.array([e2 for e1, e2, ep, rel, t, e1d, e2d, s, did in batch])
        _ep = np.array([ep for e1, e2, ep, rel, t, e1d, e2d, s, did in batch])
        _rel = np.array([rel for e1, e2, ep, rel, t, e1d, e2d, s, did in batch])
        _tokens = np.array([t for e1, e2, ep, rel, t, e1d, e2d, s, did in batch])
        _e1d = np.array([e1d for e1, e2, ep, rel, t, e1d, e2d, s, did in batch])
        _e2d = np.array([e2d for e1, e2, ep, rel, t, e1d, e2d, s, did in batch])
        _seq_len = np.array([s for e1, e2, ep, rel, t, e1d, e2d, s, did in batch])
        _doc_ids = np.array([did for e1, e2, ep, rel, t, e1d, e2d, s, did in batch])
        batch = (_e1, _e2, _ep, _rel, _tokens, _e1d, _e2d, _seq_len, _doc_ids)
        if sum(self._bucket_probs.itervalues()) == 0:
            self.reset_batch_pointer()
        return batch

    def reset_batch_pointer(self):
        # shuffle each bucket
        for bucket in self.data.itervalues():
            shuffle(bucket)
        self.epoch += 1
        self.step = 0.
        # print('\nStarting epoch %d' % self.epoch)
        self._starts = {i: 0 for i in self.data.iterkeys()}
        self._ends = {i: min(self._batch_size, len(examples)) for i, examples in self.data.iteritems()}
        self._bucket_probs = {i: len(l) for (i, l) in self.data.iteritems()}

    def select_bucket(self):
        buckets, weights = zip(*[(i, p) for i, p in self._bucket_probs.iteritems() if p > 0])
        total = float(sum(weights))
        probs = [w / total for w in weights]
        bucket = np.random.choice(buckets, p=probs)
        return bucket


def ner_example_parser(filename_queue):
    reader = tf.TFRecordReader()
    key, record_string = reader.read(filename_queue)

    # Define how to parse the example
    context_features = {
        'seq_len': tf.FixedLenFeature([], tf.int64),
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "ner_labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "entities": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=record_string,
                                                                       context_features=context_features,
                                                                       sequence_features=sequence_features)
    tokens = sequence_parsed['tokens']
    ner_labels = sequence_parsed['ner_labels']
    entities = sequence_parsed['entities']
    seq_len = context_parsed['seq_len']

    return [tokens, ner_labels, entities, seq_len]


class NERInMemoryBatcher(InMemoryBatcher):
    def example_parser(self, filename_queue):
        return ner_example_parser(filename_queue)

    def load_all_data(self, sess, max_batches=-1):
        '''
        load batches to memory for shuffling and dynamic padding
        '''
        batch_num = 0
        samples = 0
        start_time = time.time()
        print ('Loading data from %s with batch size: %d' % (self.in_file, self._batch_size))
        try:
            while True:
                batch = sess.run(self.next_batch_op)
                tokens, ner_labels, entities, seq_len = batch
                samples += ner_labels.shape[0]
                self.data[tokens.shape[1]].append(batch)
                batch_num += 1
                sys.stdout.write('\rLoading batch: %d' % batch_num)
                sys.stdout.flush()
        except Exception as e:
            print('')
        for seq_len, batches in self.data.iteritems():
            self.data[seq_len] = [tuple((t[i], n[i], e[i], s[i]))
                                  for (t, n, e, s) in batches
                                  for i in range(s.shape[0])]
        self.reset_batch_pointer()
        end_time = time.time()
        print('Done, loaded %d samples in %5.2f seconds' % (samples, (end_time-start_time)))
        return batch_num

    def next_batch(self, sess):
        # select bucket to create batch from
        self.step += 1
        bucket = self.select_bucket()
        batch = self.data[bucket][self._starts[bucket]:self._ends[bucket]]
        # update pointers
        self._starts[bucket] = self._ends[bucket]
        self._ends[bucket] = min(self._ends[bucket] + self._batch_size, len(self.data[bucket]))
        self._bucket_probs[bucket] = max(0, len(self.data[bucket]) - self._starts[bucket])

        _tokens = np.array([t for t, n, e, s in batch])
        _labels = np.array([n for t, n, e, s in batch])
        _entities = np.array([n for t, n, e, s in batch])
        _seq_len = np.array([s for t, n, e, s in batch])
        batch = (_tokens, _labels, _entities, _seq_len)
        if sum(self._bucket_probs.itervalues()) == 0:
            self.reset_batch_pointer()
        return batch


class NERBatcher(Batcher):
    def example_parser(self, filename_queue):
        return ner_example_parser(filename_queue)