#!/usr/bin/env bash

input_dir=/home/pat/data/biocreative/BC_VI_Task5/
# replace infrequent tokens with <UNK>
min_count=5
max_len=500000

processed_dir=${input_dir}/processed_genia_sentence
mkdir -p ${processed_dir}
word_piece_vocab=${processed_dir}/bpe.vocab
proto_dir=${processed_dir}/protos

echo "converting initial data to pubtator format"
python ${CDR_IE_ROOT}/src/processing/utils/biocreative_VI_task5.py --input_dir ${input_dir}/chemprot_training --split training
python ${CDR_IE_ROOT}/src/processing/utils/biocreative_VI_task5.py --input_dir ${input_dir}/chemprot_development --split development
python ${CDR_IE_ROOT}/src/processing/utils/biocreative_VI_task5.py --input_dir ${input_dir}/chemprot_test --split test

# process train, dev, and test data
echo "Processing Training data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/chemprot_training/chemprot_training_pubtator.tsv --output_dir ${processed_dir} --output_file_suffix CDR_train.txt --max_seq ${max_len} --full_abstract --encoding utf-8 --export_all_eps

echo "Processing Dev data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/chemprot_development/chemprot_development_pubtator.tsv --output_dir ${processed_dir} --output_file_suffix CDR_dev.txt --max_seq ${max_len} --full_abstract --encoding utf-8 --export_all_eps

#echo "Processing Test data"
#python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/chemprot_test/chemprot_test_pubtator.tsv --output_dir ${processed_dir} --output_file_suffix CDR_test.txt --max_seq ${max_len} --full_abstract True --encoding utf-8 --export_all_eps True --export_negatives True --max_distance=${max_distance}

# convert processed data to tensorflow protobufs
python ${CDR_IE_ROOT}/src/processing/labled_tsv_to_tfrecords_single_sentences.py --text_in_files ${processed_dir}/\*tive_\*CDR\* --out_dir ${proto_dir} --max_len ${max_len} --num_threads 10 --multiple_mentions --tsv_format --min_count ${min_count}

# rename dev files to match train regex
mv ${proto_dir}/negative_0_CDR_dev.txt.proto ${proto_dir}/negative_0_CDR_train_dev.txt.proto
mv ${proto_dir}/positive_0_CDR_dev.txt.proto ${proto_dir}/positive_0_CDR_train_dev.txt.proto

# convert ner data to tf protos
python ${CDR_IE_ROOT}/src/processing/ner_to_tfrecords.py --in_files ${processed_dir}/ner_\* --out_dir ${proto_dir} --load_vocab ${proto_dir} --num_threads 5