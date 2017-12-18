import argparse
import codecs
from collections import defaultdict
'''

'''

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--positive_input_file', required=True, help='input file in 13col tsv')
parser.add_argument('-n', '--negative_input_file', required=True, help='input file in 13col tsv')
parser.add_argument('-m', '--mesh_file', required=True, help='mesh file to get heriarchy from')
parser.add_argument('-o', '--output_file', required=True, help='write results to this file')

args = parser.parse_args()


# read in mesh heirarchy
ent_tree_map = defaultdict(list)
with codecs.open(args.mesh_file,'r', encoding='utf-16-le') as f:
    # lines = [[s.decode('utf-16le') for s in l.rstrip().split('\t')] for i, l in enumerate(f) if i > 0]
    lines = [l.rstrip().split('\t') for i, l in enumerate(f) if i > 0]
    [ent_tree_map[l[1]].append(l[0]) for l in lines]

# read in positive input file and organize by document
print('Loading positive examples from %s' % args.positive_input_file)
pos_doc_examples = defaultdict(list)
with open(args.positive_input_file, 'r') as f:
    lines = [l.strip().split('\t') for l in f]
    unfilitered_pos_count = len(lines)
    [pos_doc_examples[l[10]].append(l) for l in lines]

# read in negative input file and organize by document
print('Loading negative examples from %s' % args.negative_input_file)
neg_doc_examples = defaultdict(list)
with open(args.negative_input_file, 'r') as f:
    lines = [l.strip().split('\t') for l in f]
    unfilitered_neg_count = len(lines)
    [neg_doc_examples[l[10]].append(l) for l in lines]

#iterate over docs
hypo_count = 0
negative_count = 0
with open(args.output_file, 'w') as out_f:
    for doc_id in neg_doc_examples.keys():
        # get nodes for all the positive diseases
        pos_e2_examples = [(pos_node, pe) for pe in pos_doc_examples[doc_id]
                           for pos_node in ent_tree_map[pe[5]]]
        pos_e1_examples = [(pos_node, pe) for pe in pos_doc_examples[doc_id]
                           for pos_node in ent_tree_map[pe[0]]]
        filtered_neg_exampled = []
        for ne in neg_doc_examples[doc_id]:
            neg_e1 = ne[0]
            neg_e2 = ne[5]
            example_hyponyms = 0
            for neg_node in ent_tree_map[ne[5]]:
                hyponyms = [pos_node for pos_node, pe in pos_e2_examples
                            if neg_node in pos_node and neg_e1 == pe[0]] \
                           + [pos_node for pos_node, pe in pos_e1_examples
                              if neg_node in pos_node and neg_e2 == pe[5]]
                example_hyponyms += len(hyponyms)
            if example_hyponyms == 0:
                out_f.write('\t'.join(ne)+'\n')
                negative_count += 1
            else:
                hypo_count += example_hyponyms

print('Mesh entities: %d' % len(ent_tree_map))
print('Positive Docs: %d' % len(pos_doc_examples))
print('Negative Docs: %d' % len(neg_doc_examples))
print('Positive Count: %d   Initial Negative Count: %d   Final Negative Count: %d   Hyponyms: %d' %
      (unfilitered_pos_count, unfilitered_neg_count, negative_count, hypo_count))
