#!/usr/bin/env bash

CTD_DIR=${CDR_IE_ROOT}/data/ctd
PUBTATOR_FILE=bioconcepts2pubtator_offsets.gz
MAX_LEN=500
MIN_COUNT=5
vocab_size=50000
FILE_SUFFIX="${vocab_size}_vocab"
word_piece_vocab="${CTD_DIR}/word_piece_vocabs/word_pieces_${vocab_size}.txt"
PROCESSED_DIR=${CTD_DIR}/CTD_${vocab_size}


mkdir -p ${PROCESSED_DIR}

if [ ! -f ${CTD_DIR}/${PUBTATOR_FILE} ]; then
    echo "Downloading Pubtator dump"
    # download the pubtator file NOTE: this is 10GB+
    wget ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator/${PUBTATOR_FILE} -P ${CTD_DIR}
fi

if [ ! -f ${CTD_DIR}/CTD_all_entities_pubtator_interactions.gz ]; then
    echo "Filtering to keep only pubmed articles with currated relations"
    python ${CDR_IE_ROOT}/src/processing/utils/filter_pubtator_keep_format.py -b ${CTD_DIR}/${PUBTATOR_FILE} -r ${CTD_DIR}/curated_ctd_interactions.gz -o ${CTD_DIR}/CTD_all_entities_pubtator_interactions.gz
fi

if [ ! -f ${word_piece_vocab} ]; then
    echo "generating word-piece vocab with ${vocab_size} tokens"
    python ${CDR_IE_ROOT}/src/processing/utils/learn_bpe.py -i <(less ${CTD_DIR}/CTD_all_entities_pubtator_interactions.gz | grep -e '|t|' -e '|a|' | sed 's/^[0-9]\+|[ta]|//') -o ${word_piece_vocab} -s ${vocab_size}
fi

echo "Converting data from pubtator to tsv format"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py \
--input_file ${CTD_DIR}/CTD_all_entities_pubtator_interactions.gz \
--output_dir ${PROCESSED_DIR} \
--output_file_suffix ${FILE_SUFFIX} --max_seq 5000000 --full_abstract True -x True \
--word_piece_codes ${word_piece_vocab}

# split the data up - hacky
cp ${CDR_IE_ROOT}/bin/process_CTD/process_ctd_dir.sh ${PROCESSED_DIR}
cd ${PROCESSED_DIR}
./process_ctd_dir.sh ${FILE_SUFFIX}
cd ${CDR_IE_ROOT}

echo "converting data to TF protos"
python src/processing/labled_tsv_to_tfrecords.py --text_in_files ${PROCESSED_DIR}/shards/\*tive\* --out_dir ${PROCESSED_DIR}/protos --max_len ${MAX_LEN} --num_threads 30 --multiple_mentions --tsv_format --min_count ${MIN_COUNT}

echo "converting NER data to TF protos"
python src/processing/ner_to_tfrecords.py --in_file ${PROCESSED_DIR}/ner_${FILE_SUFFIX}\* --out_dir ${PROCESSED_DIR}/protos/ --load_vocab ${PROCESSED_DIR}/protos/ --num_threads 5