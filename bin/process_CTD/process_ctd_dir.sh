#!/bin/bash

FILE_SUFFIX=$1
PMID_SPLITS=${CDR_IE_ROOT}/data/ctd/pubmed_split_lengths.txt
MAX_LEN=500
MIN_LEN=50

mkdir -p shards ner_shards split_rel_counts

echo "map relations to smaller set"
awk -F '\t' ' NR==FNR{if($2 != ""){ map[$1] = $2}; next }{if($12 in map){rel=map[$12]; print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10"\t"$11"\t"rel"\t"$13}}' ${CDR_IE_ROOT}/data/ctd/relation_map.txt positive_0_${FILE_SUFFIX} > positive_0_${FILE_SUFFIX}_mapped

echo "seperate data into train dev test"
for in_file in positive_0_${FILE_SUFFIX}_mapped negative_0_${FILE_SUFFIX}; do
  label=`echo $in_file | cut -d '_' -f 1`
  for split in train dev test; do 
    echo "$label $split $MIN_LEN $MAX_LEN";
    awk -F'\t' -vmin_len=${MIN_LEN} -vmax_len=${MAX_LEN} -vs=${split} 'NR==FNR{if($3 <= max_len && $3 >= min_len){pmid[$1]=$2}; next }{if(pmid[$11]==s){{print}}}' ${PMID_SPLITS} ${in_file} | shuf > ${label}_${FILE_SUFFIX}_${split};
  done;
done

echo "shard data to be processed"
for label in positive negative; do \
  for split in train dev test; do \
    f="${label}_${FILE_SUFFIX}_${split}"; \
    echo $f; \
    split --lines 10000 -d ${f} shards/${f}; \
  done; \
done

echo "seperate ner data"
for split in train dev test; do 
  echo ${split};
  awk -F'\t' -vmin_len=${MIN_LEN} -vmax_len=${MAX_LEN} -vs=${split} 'NR==FNR{if($3 <= max_len && $3 >= min_len){pmid[$1]=$2}; next }{if(pmid[$4]==s || NF==0){print}}' ${PMID_SPLITS} ner_${FILE_SUFFIX}  | cat -s | sed '1{/^$/d}'> ner_${FILE_SUFFIX}_${split} ;
done

# get split relation counts
for split in train dev test; do awk -F $'\t' '{c[$12]+=1}END{for(r in c) print r"\t"c[r]}' positive_${FILE_SUFFIX}_${split} | sort > split_rel_counts/${split}; done
join <(join split_rel_counts/train split_rel_counts/dev) split_rel_counts/test  > split_rel_counts/all 

# calculate label weights
cat positive_${FILE_SUFFIX}_train |awk -F$'\t' '{c[$12]+=1}END{max=0;for(i in c){if (c[i] > max){max=c[i]}}; for(i in c){print i"\t"1+(max-c[i])/c[i]}}' | sort -nk2 > label_weights.txt
