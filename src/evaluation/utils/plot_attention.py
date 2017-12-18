from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import string
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--numpy_file', required=True, default='attention_weights.npz',
                    help='file containing numpy dictionary of values')
parser.add_argument('-k', '--key_file', required=True, default='attention_weights.txt',
                    help='text file containing corresponding keys')
parser.add_argument('-n', '--ner_file', required=True, help='file with ner annotations')
parser.add_argument('-o', '--out_dir', required=True, help='dir to write all outputs to')
parser.add_argument('-m', '--max_len', type=int, default=True, help='only plot documents up to this length')
args = parser.parse_args()


# index of deepest layer
max_layer = 1
# colors for highlighting entities & relations labels
# colors_map = {0: 'black', 1: 'red', 2: 'blue'}
colors_map = {'O': 'black', 'Chemical': 'red', 'Disease': 'blue',
              'Gene': 'green', 'Species': 'cyan', 'Mutation': 'magenta'}
data = np.load(args.numpy_file)
save_dir = args.out_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def sorted_alphanum(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# read in ner annotations that we will use to color the tokens
doc_label_map = {}
with open(args.ner_file, 'r') as f:
    current_tokens = []
    current_labels = []
    current_entities = []
    for line in f:
        line = line.strip()
        if line:
            token, label, entity, did = line.split('\t')
            current_tokens.append(token)
            label = label[2:] if (label.startswith('B-') or label.startswith('I-')) else label
            label = 'Mutation' if (label.endswith('Mutation') or label.endswith('SNP')) else label
            label = 'O' if label not in colors_map else label
            current_labels.append(label)
            current_entities.append(entity)
        # end of doc
        else:
            doc_label_map[' '.join(current_tokens)] = current_labels
            current_tokens = []
            current_labels = []
            current_entities = []


lines = map(lambda x: x.split('\t'), open(args.key_file, 'r').readlines())
docs = []
current_doc = []
current_batch = current_example = current_tok = "0"
for line in lines:
    _, this_batch, this_example, this_tok = re.split("[a-z_]*", line[0])
    if this_example != current_example:
        docs.append(map(list, zip(*current_doc)))
        current_doc = [map(string.strip, line[1:])]
        current_example = this_example
    else:
        if line[1] != "<PAD>":
            current_doc.append(map(string.strip, line[1:]))

docs.append(map(list, zip(*current_doc)))


batch_size = data[data.files[0]].shape[0]

# only plot a subsample of this many documents
sample = False
num_samples = 20
if sample:
    total_num = batch_size * len(data.files)
    samples = np.random.choice(total_num, num_samples, replace=False)
    print("Plotting samples: %s" % (' '.join(map(str, sorted(samples)))))

# For each batch+layer
batch_sum = 0
fig, ax = plt.subplots()
for arr_name in sorted_alphanum(data.files):
    print("Processing %s" % arr_name)

    split_name = re.split("[a-z_]*", arr_name)
    batch = int(split_name[1])
    layer = int(split_name[2])

    idx_in_batch = 0
    # For each element in the batch (one layer)
    # if layer == max_layer and batch > 0:
    for b_i, arrays in enumerate(data[arr_name]):
        doc_idx = batch_sum + b_i
        if not sample or doc_idx in samples:
            if sample:
                print("Taking batch: %d, doc: %d, layer: %d" % (batch, doc_idx, layer))

            width = arrays.shape[-1]
            doc = docs[doc_idx]
            words = doc[0]
            doc_str = ' '.join(words)
            if doc_str in doc_label_map:
                ner_labels = doc_label_map[doc_str]
                # e1 = np.array(map(int, doc[1]))
                # e2 = np.array(map(int, doc[2]))
                doc_len = len(words)
                if doc_len <= args.max_len:
                    # tick_colors = map(colors_map.get, (2 * e1 + e2)[:doc_len])
                    tick_colors = map(colors_map.get, ner_labels)
                    # words = ['%s__%s' % (w, e) if e != '-1' else w for w, e in zip(words, entities)]

                    # For each attention head
                    for head, arr in enumerate(arrays):

                        name = "doc%d_layer%d_head%d_%d" % (doc_idx, layer, head, doc_len)

                        ax.set_title(name, fontsize=8)
                        # axis 1 of arr sums to 1
                        res = ax.imshow(arr[:doc_len, :doc_len], cmap=plt.cm.viridis, interpolation=None)
                        ax.set_xticks(range(doc_len))
                        ax.set_yticks(range(doc_len))
                        ax.set_xticklabels(words, rotation=75, fontsize=2)
                        ax.set_yticklabels(words, fontsize=2)
                        for t, c in zip(ax.get_xticklabels(), tick_colors):
                            t.set_color(c)
                        for t, c in zip(ax.get_yticklabels(), tick_colors):
                            t.set_color(c)

                        fig.tight_layout()
                        fig.savefig(os.path.join(save_dir, name + ".pdf"))
                        ax.clear()

    if layer == max_layer:
        batch_sum += batch_size