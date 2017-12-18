#!/usr/bin/env bash

input_dir=/home/pat/data/biocreative/BC_VI_Task5/
CTD_dir=${CDR_IE_ROOT}/data/ctd
vocab_size=12500
# replace infrequent tokens with <UNK>
min_count=5
max_len=500000

processed_dir=${input_dir}/processed_${vocab_size}_ctd
mkdir -p ${processed_dir}
word_piece_vocab=${processed_dir}/bpe.vocab
proto_dir=${processed_dir}/protos


echo "converting initial data to pubtator format"
python ${CDR_IE_ROOT}/src/processing/utils/biocreative_VI_task5.py --input_dir ${input_dir}/chemprot_training --split training
python ${CDR_IE_ROOT}/src/processing/utils/biocreative_VI_task5.py --input_dir ${input_dir}/chemprot_development --split development
python ${CDR_IE_ROOT}/src/processing/utils/biocreative_VI_task5.py --input_dir ${input_dir}/chemprot_test --split test

echo "Getting ctd data with chem_gene relations"
zcat ${CTD_dir}/CTD_all_entities_pubtator_interactions.gz |grep "chem_gene" | awk -F'\t' '{m[$1"\t"$2]+=1}END{for(i in m){ print(i)}}' > ${processed_dir}/chem_gene_docs.txt
awk -F '\t' ' NR==FNR{ entities[$1] = 1; next }{split($1, a, "|"); if(NF==0){print}else if(a[1] in entities){print}}' ${processed_dir}/chem_gene_docs.txt <(zcat ${CTD_dir}/CTD_all_entities_pubtator_interactions.gz) | cat -s | sed '1{/^$/d}' | gzip > ${processed_dir}/chem_gene_pubtator.gz

echo "generating word piece vocab"
awk -F $'\t' '{print $2"\n"$3}' ${input_dir}/chemprot_training/chemprot_training_abstracts.tsv |gzip > ${processed_dir}/text_only.gz
grep -e "|a|" -e "|t|" <(zcat ${processed_dir}/chem_gene_pubtator.gz) | cut -d "|" -f 3- | gzip >> ${processed_dir}/text_only.gz
python ${CDR_IE_ROOT}/src/processing/utils/learn_bpe.py -i <(zcat ${processed_dir}/text_only.gz) -o ${word_piece_vocab} -s 10000

# process train, dev, and test data
echo "Processing Training data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/chemprot_training/chemprot_training_pubtator.tsv --output_dir ${processed_dir} --output_file_suffix CDR_train.txt --max_seq ${max_len} --full_abstract --word_piece_codes ${word_piece_vocab} --encoding utf-8 --export_all_eps

echo "Processing Dev data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/chemprot_development/chemprot_development_pubtator.tsv --output_dir ${processed_dir} --output_file_suffix CDR_dev.txt --max_seq ${max_len} --full_abstract --word_piece_codes ${word_piece_vocab} --encoding utf-8 --export_all_eps

echo "Processing Test data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/chemprot_test/chemprot_test_pubtator.tsv --output_dir ${processed_dir} --output_file_suffix CDR_test.txt --max_seq ${max_len} --full_abstract --word_piece_codes ${word_piece_vocab} --encoding utf-8 --export_all_eps --export_negatives

echo "processing ctd data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${processed_dir}/chem_gene_pubtator.gz --output_dir ${processed_dir} --output_file_suffix ctd_chem_gene_train.txt --max_seq 500 --full_abstract --word_piece_codes ${word_piece_vocab}  --export_all_eps
# map relations
awk -F '\t' ' NR==FNR{if($2 != ""){ map[$1] = $2}; next }{if($12 in map){rel=map[$12]; print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10"\t"$11"\t"rel"\t"$13}}' ${CDR_IE_ROOT}/data/ctd/relation_map.txt ${processed_dir}/positive_0_ctd_chem_gene_train.txt > ${processed_dir}/positive_0_ctd_chem_gene_train_mapped.txt


# convert processed data to tensorflow protobufs
python ${CDR_IE_ROOT}/src/processing/labled_tsv_to_tfrecords.py --text_in_files ${processed_dir}/\*tive_\* --out_dir ${proto_dir} --max_len ${max_len} --num_threads 10 --multiple_mentions --tsv_format --min_count ${min_count}

## rename dev files to match train regex
#mv ${proto_dir}/negative_0_CDR_dev.txt.proto ${proto_dir}/negative_0_CDR_train_dev.txt.proto
#mv ${proto_dir}/positive_0_CDR_dev.txt.proto ${proto_dir}/positive_0_CDR_train_dev.txt.proto

# convert ner data to tf protos
python ${CDR_IE_ROOT}/src/processing/ner_to_tfrecords.py --in_files ${processed_dir}/ner_\* --out_dir ${proto_dir} --load_vocab ${proto_dir} --num_threads 5