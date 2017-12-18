import argparse
import subprocess
import sys
import os
import glob
from operator import itemgetter


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--predictions', required=True, help='prediction files')
parser.add_argument('-o', '--output_file', required=True, help='write results to this file')
parser.add_argument('-v', '--vocab_file', required=True, help='vocab file (probably data_dir/protos/rel.txt)')
parser.add_argument('-e', '--eval_script', required=True, help='eval_script')
parser.add_argument('-m', '--max_score', default=False, help='only consider max scoring label for each prediction')
parser.add_argument('-f', '--filter_file',  help='same format as gold file, only consider predictions in this filter')
parser.add_argument('-g', '--gold_file',  help='gold annotation file')
parser.add_argument('-j', '--jar_file', default='/home/pat/data/biocreative/BC_VI_Task5/bc6chemprot_eval.jar',
                    help='gold annotation file')
parser.add_argument('-n', '--null_labels', default='Null', help='gold annotation file')
parser.add_argument('-t', '--thresholds', help='pre-trained thresholds')
args = parser.parse_args()

null_labels = set(args.null_labels.split(','))
with open(args.vocab_file, 'r') as f:
    parts = [l.strip().split('\t') for l in f]
    label_map = {int(_id): label for label, _id in parts}
    threshold_map = {label: 1.0 for label, _id in parts if label not in null_labels}

with open(args.filter_file, 'r') as f:
    parts = [l.strip().split('\t') for l in f]
    filter_set = set()
    for did, _, e1, e2 in parts:
        filter_set.add((did, e1, e2))

print('Reading in prediction files from: %s' % args.predictions)
all_predictions = {}
file_names = glob.glob(args.predictions)
num_files = float(len(file_names))

for i, pred_file in enumerate(file_names):
    if pred_file:
        with open(pred_file, 'r') as f:
            parts = [l.strip().split('\t') for l in f]
            parts = [p for p in parts if len(p) == 4]
            file_predictions = {(did, e1, e2): [(label_map[label_idx], float(prob))
                                                for label_idx, prob in enumerate(predictions.split(':'))]
                                for did, e1, e2, predictions in parts if (did, e1, e2) in filter_set}

            # only keep max prediction
            if args.max_score:
                file_predictions = {(did, e1, e2): [max(preds, key=itemgetter(1))]
                                    for (did, e1, e2), preds in file_predictions.iteritems()}
            if i == 0:
                all_predictions = file_predictions
            else:
                for key, pred_list in file_predictions.iteritems():
                    if key in all_predictions:
                        current_preds = all_predictions[key]
                        file_pred_map = {l: p for l, p in pred_list}
                        all_predictions[key] = [(l, p+file_pred_map[l]) if l in file_pred_map else (l, p)
                                                for j, (l, p) in enumerate(current_preds)]
                    else:
                        all_predictions[key] = pred_list
# average the preds
predictions = {key: [(l, p/num_files) for l, p in pred_list] for key, pred_list in all_predictions.iteritems()}

# tune thresholds
if args.gold_file:
    print('Tuning thresholds')
    FNULL = open(os.devnull, 'w')
    best_score = 0.0
    for i in range(2):
        print('Iteration: %d' % i)
        for tune_label in threshold_map.iterkeys():
            best_threshold = threshold_map[tune_label]
            for threshold in [0.0, 0.025, .1, .2, .3, .4, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .975]:
                threshold_map[tune_label] = threshold
                # write thresholded results to file
                with open(args.output_file, 'w') as out_f:
                    out_str = ['%s\t%s\t%s\t%s\n' % (did, label, e1, e2)
                               for (did, e1, e2), preds in predictions.iteritems()
                               for label, score in preds
                               if label not in null_labels and score > threshold_map[label]]
                    out_f.write(''.join(out_str))

                # get score
                cmd = 'java -cp %s org.biocreative.tasks.chemprot.main.Main %s %s ' \
                      % (args.jar_file, args.output_file, args.gold_file)
                subprocess.call(cmd, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
                # read in results
                with open('out/eval.txt', 'r') as f:
                    score = float([l for l in f][-1].split(' ')[-1])

                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            print('Label: %s  Threshold: %2.3f  Score: %2.3f' % (tune_label, best_threshold, best_score))
            threshold_map[tune_label] = best_threshold

    # write results using best threshold for each label
    print(threshold_map)
    with open(args.output_file, 'w') as out_f:
        out_str = ['%s\t%s\t%s\t%s\n' % (did, label, e1, e2)
                   for (did, e1, e2), preds in predictions.iteritems()
                   for label, score in preds
                   if label not in null_labels and score >= threshold_map[label]]
        out_f.write(''.join(out_str))

    # get score
    cmd = 'java -cp %s org.biocreative.tasks.chemprot.main.Main %s %s ' \
          % (args.jar_file, args.output_file, args.gold_file)
    subprocess.call(cmd, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    # read in results
    with open('out/eval.txt', 'r') as f:
        score = float([l for l in f][-1].split(' ')[-1])
        print(score)

# use learned thresholds to export prediction file
elif args.thresholds:
    # print(args.thresholds)
    parts = [t.split(': ') for t in args.thresholds.split(',')]
    threshold_map = {label.strip().replace('\'', ''): float(val.strip()) for label, val in parts}
    print(threshold_map)
    # write thresholded results to file
    with open(args.output_file, 'w') as out_f:
        out_str = ['%s\t%s\t%s\t%s\n' % (did, label, e1, e2)
                   for (did, e1, e2), preds in predictions.iteritems()
                   for label, score in preds
                   if label not in null_labels and score > threshold_map[label]]
        out_f.write(''.join(out_str))

else:
    print('Must supply gold file or prelearned thresholds')
