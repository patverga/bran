import codecs
import argparse
from collections import defaultdict
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', required=True, help='input dir')
parser.add_argument('-s', '--split', required=True, help='training, development, or testing')
parser.add_argument('-f', '--filter_extra_relations', default=True, type=bool, help='only keep relations evaluated in task')
parser.add_argument('-e', '--extra_suffix', default="", help='add addditional suffix to filename')
args = parser.parse_args()

root_dir = args.input_dir
split = args.split

# optionally set non evaluated relations to NULL
filter_relations = set(['CPR:0', 'CPR:1', 'CPR:2', 'CPR:7', 'CPR:8', 'CPR:10']) \
    if args.filter_extra_relations else set()

# read in title / abstracts
with codecs.open('%s/%s_%s_%s%s.tsv' % (root_dir, 'chemprot', split, 'abstracts', args.extra_suffix), 'r', 'utf-8') as f:
    lines = [l.strip().split('\t') for l in f]
    text = {pmid: (title, abstract) for pmid, title, abstract in lines}

# read in entities seperate by pubmed article
doc_entities = defaultdict(list)
with codecs.open('%s/%s_%s_%s%s.tsv' % (root_dir, 'chemprot', split, 'entities', args.extra_suffix), 'r', 'utf-8') as f:
    lines = [l.strip().split('\t') for l in f]
    for pmid, e_id, e_type, start, end, e_str in lines:
        # only have a single gene type
        e_type = 'Gene' if e_type.startswith('GENE') else 'Chemical'
        doc_entities[pmid].append((e_id, e_type, start, end, e_str))

# read in relations, seperate by pubmed article
doc_relations = defaultdict(list)
relation_fname = '%s/%s_%s_%s' % (root_dir, 'chemprot', split, 'gold_standard.tsv')
if os.path.isfile(relation_fname):
    with codecs.open(relation_fname, 'r', 'utf-8') as f:
        lines = [l.strip().split('\t') for l in f]
        for pmid, r_type, arg1, arg2 in lines:
            if r_type not in filter_relations:
                arg1 = arg1.split(':')[1]
                arg2 = arg2.split(':')[1]
                doc_relations[pmid].append((r_type, arg1, arg2))

# export as pubtator format
with codecs.open('%s/%s_%s_%s' % (root_dir, 'chemprot', split, 'pubtator.tsv'), 'w', 'utf-8') as f:
    for pmid, (title, abstract) in text.iteritems():
        # export text
        f.write('%s|t|%s\n' % (pmid, title))
        f.write('%s|a|%s\n' % (pmid, abstract))
        # export entities
        for e_id, e_type, start, end, e_str in doc_entities[pmid]:
            f.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (pmid, start, end, e_str, e_type, e_id))
        # export relations
        for r_type, arg1, arg2 in doc_relations[pmid]:
            f.write('%s\t%s\t%s\t%s\n' % (pmid, r_type, arg1, arg2))
        f.write('\n')
