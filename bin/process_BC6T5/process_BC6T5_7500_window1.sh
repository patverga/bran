#!/usr/bin/env bash

input_dir=/home/pat/data/biocreative/ChemProt_Corpus/
# replace infrequent tokens with <UNK>
min_count=5
max_len=500000
vocab_size=7500
processed_dir=${input_dir}/processed_7500_window1
mkdir -p ${processed_dir}
word_piece_vocab=${processed_dir}/bpe.vocab
proto_dir=${processed_dir}/protos

echo "converting initial data to pubtator format"
python ${CDR_IE_ROOT}/src/processing/utils/biocreative_VI_task5.py --input_dir ${input_dir}/chemprot_training --split training --filter_extra_relations True
python ${CDR_IE_ROOT}/src/processing/utils/biocreative_VI_task5.py --input_dir ${input_dir}/chemprot_development --split development --filter_extra_relations True
python ${CDR_IE_ROOT}/src/processing/utils/biocreative_VI_task5.py --input_dir ${input_dir}/chemprot_test_gs --split test --filter_extra_relations True --extra_suffix _gs

echo "generating word-piece vocab with ${vocab_size} tokens"
python ${CDR_IE_ROOT}/src/processing/utils/learn_bpe.py -i <(awk -F $'\t' '{print $2"\n"$3}' ${input_dir}/chemprot_training/chemprot_training_abstracts.tsv) -o ${word_piece_vocab} -s ${vocab_size}


# process train, dev, and test data
echo "Processing Training data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/chemprot_training/chemprot_training_pubtator.tsv --output_dir ${processed_dir} --output_file_suffix CDR_train.txt --max_seq ${max_len} --full_abstract True --encoding utf-8 --export_all_eps True --word_piece_codes ${word_piece_vocab}

echo "Processing Dev data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/chemprot_development/chemprot_development_pubtator.tsv --output_dir ${processed_dir} --output_file_suffix CDR_dev.txt --max_seq ${max_len} --full_abstract True --encoding utf-8 --export_all_eps True --word_piece_codes ${word_piece_vocab}

echo "Processing Test data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/chemprot_test_gs/chemprot_test_pubtator.tsv --output_dir ${processed_dir} --output_file_suffix CDR_test.txt --max_seq ${max_len} --full_abstract True --encoding utf-8 --export_all_eps True --word_piece_codes ${word_piece_vocab}

echo "Converting processed data to tensorflow protobufs"
python ${CDR_IE_ROOT}/src/processing/labled_tsv_to_tfrecords_single_sentences.py --text_in_files ${processed_dir}/\*tive_\*CDR_train.txt --out_dir ${proto_dir} --max_len ${max_len} --num_threads 10 --multiple_mentions --tsv_format --min_count ${min_count} --sentence_window 1
python ${CDR_IE_ROOT}/src/processing/labled_tsv_to_tfrecords_single_sentences.py --text_in_files ${processed_dir}/\*tive_\*CDR_\*e\*.txt --out_dir ${proto_dir} --max_len ${max_len} --num_threads 10 --multiple_mentions --tsv_format --min_count ${min_count} --load_vocab ${proto_dir} --sentence_window 1


# rename dev files to match train regex
mv ${proto_dir}/negative_0_CDR_dev.txt.proto ${proto_dir}/negative_0_CDR_train_dev.txt.proto
mv ${proto_dir}/positive_0_CDR_dev.txt.proto ${proto_dir}/positive_0_CDR_train_dev.txt.proto

# convert ner data to tf protos
python ${CDR_IE_ROOT}/src/processing/ner_to_tfrecords.py --in_files ${processed_dir}/ner_\* --out_dir ${proto_dir} --load_vocab ${proto_dir} --num_threads 5