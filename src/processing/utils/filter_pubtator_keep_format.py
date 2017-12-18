import argparse
import gzip
import sys
import itertools
'''

'''

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bio_pub', required=True, help='pubtator bio offset file')
parser.add_argument('-o', '--output_file', required=True, help='write results to this file')
parser.add_argument('-n', '--entity_filter', help='single column file of entities to keep')
parser.add_argument('-e', '--entity_pair_filter', help='2 column tsv of entity pairs to export')
parser.add_argument('-p', '--pubmed_filter', help='only export these pubmed ids')
parser.add_argument('-r', '--relation_file', help='4 col tsv containing relations to add to output. '
                                                  '[e1 \t e2 \t relation \t docid]')

args = parser.parse_args()

current_annotations = []
current_pub = ''
line_num = 0
done = False
valid_annotations = 0
valid_pubs = 0
exported_annotations = 0
exported_abstracts = 0
total_abstracts = 0


print ('Reading in filter files')
pubmed_filter, entity_filter, ep_filter, doc_relation_map = None, None, None, None
if args.pubmed_filter:
    with open(args.pubmed_filter) as f:
        pubmed_filter = set([l.strip() for l in f])
if args.entity_filter:
    with open(args.entity_filter) as f:
        entity_filter = set([l.strip() for l in f])
if args.entity_pair_filter:
    with open(args.entity_pair_filter) as f:
        ep_filter = set([(l.strip().split('\t')[0], l.strip().split('\t')[1]) for l in f])

if args.relation_file:
    print('Reading in relation file %s' % args.relation_file)
    with (gzip.open(args.relation_file, 'rb') if args.relation_file.endswith('gz')
          else open(args.relation_file, 'r')) as rel_file:
        doc_relation_map = {(doc_id, e1, e2): rel for e1, e2, rel, doc_id in [_l.strip().split() for _l in rel_file]}

with (gzip.open(args.output_file, 'wb') if args.output_file.endswith('.gz') else open(args.output_file, 'r')) as out_f:
    with (gzip.open(args.bio_pub, 'rb') if args.bio_pub.endswith('.gz') else open(args.bio_pub, 'r')) as f:
        for line in f:
            if line_num == 0:
                title = line
                doc_id = title.split('|')[0]
                abstract = f.readline()
            line_num += 1
            if line_num % 10000 == 0:
                sys.stdout.write('\rline: %dK   exported_annotations: %dK  '
                                 'exported_abstracts: %dK  total_abstracts : %dK'
                                 % (line_num/1000, exported_annotations/1000,
                                    exported_abstracts/1000, total_abstracts/1000))
                sys.stdout.flush()
            # new pub
            if len(line.strip()) == 0:
                # do something with last annotations
                if valid_annotations > 0:
                    valid_pubs += 1
                    replaced_text = []
                    last = 0
                    annotation_map = {}
                    entities_in_abstract = [_kg_id for _kg_id, _line in current_annotations]

                    if ep_filter:
                        matched_eps = [pair for pair in itertools.product(entities_in_abstract, repeat=2)
                                       if pair in ep_filter]
                    if doc_relation_map:
                        matched_relations = set(['\t'.join([doc_id, doc_relation_map[(doc_id, e1, e2)], e1, e2])
                                                 for e1, e2 in itertools.product(entities_in_abstract, repeat=2)
                                                 if (doc_id, e1, e2) in doc_relation_map])
                    # if example matches the filters or there are no filters
                    if (not pubmed_filter or pub_id in pubmed_filter) \
                            and (not ep_filter or matched_eps) \
                            and (not doc_relation_map or matched_relations):
                        exported_abstracts += 1
                        # write sentences and annotations to file
                        out_str = '%s%s' % (title, abstract)
                        exported_annotations += len(current_annotations)
                        out_str += ''.join([_line for _kg_id, _line in current_annotations])
                        out_str += '\n'.join([_line for _line in matched_relations])
                        out_f.write(out_str + '\n\n')
                total_abstracts += 1

                # reset annotations for next pub
                current_annotations = []
                valid_annotations = 0
                title = f.readline()
                doc_id = title.split('|')[0]
                abstract = f.readline()
            else:
                parts = line.strip().split('\t')
                if len(parts) == 6:
                    pub_id, start, end, mention, label, kg_id = parts
                    kg_id = kg_id.replace('MESH:', '')
                    line = line.replace('MESH:', '')
                    current_annotations.append((kg_id, line))
                    if not entity_filter or kg_id in entity_filter:
                        valid_annotations += 1

print('Done')