import argparse
import sys
import codecs
from nltk.tokenize import sent_tokenize


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', required=True, help='input file in conll format')
parser.add_argument('-o', '--output_file', required=True, help='output file in conll format')
parser.add_argument('-e', '--encoding', default='utf-8', help='input encoding')

args = parser.parse_args()


with codecs.open(args.input_file, 'r', encoding=args.encoding) as f:
    lines = [l.strip() for l in f]

tokens, labels, kgids, docids = [], [], [], []
abstract_num = 0
errors = 0
with codecs.open(args.output_file, 'w', encoding=args.encoding) as out_f:
    for line_num, line in enumerate(lines):
        if line:
            token, label, kgid, docid = line.split('\t')
            tokens.append(token)
            labels.append(label)
            kgids.append(kgid)
            docids.append(docid)
        # abstract is over
        else:
            if abstract_num % 100 == 0:
                print (abstract_num, len(tokens), len(labels), len(kgids), len(docids), errors)
            abstract = ' '.join(tokens)
            sentences = sent_tokenize(abstract, 'english')
            new_tokens = ' '.join(sentences).split(' ')
            cur_token = 0
            if len(new_tokens) != len(tokens):
                print('ERROR')
                errors += 1
            else:
                for sentence in sentences:
                    tokens = sentence.split(' ')
                    for token_num, token in enumerate(tokens):
                        label = labels[cur_token]
                        kgid = kgids[cur_token]
                        docid = docids[cur_token]
                        # last token of sentence had . attached
                        if token_num == (len(tokens)-1) and token != '.' and token.endswith('.'):
                            out_line = '%s\t%s\t%s\t%s\n%s\t%s\t%s\t%s\n' \
                                       % (token[:-1], label, kgid, docid, '.', 'O', '-1', docid)
                            out_f.write(out_line)
                        else:
                            out_line = '%s\t%s\t%s\t%s\n' % (token, label, kgid, docid)
                            out_f.write(out_line)
                        cur_token += 1

                    out_f.write('\n')
            abstract_num += 1

            tokens, labels, kgids, docids = [], [], [], []


