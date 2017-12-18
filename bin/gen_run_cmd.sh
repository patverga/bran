#!/usr/bin/env bash

conf=$1
if [ ! -e $conf ]; then
    echo "No config file specified; Exiting."
    exit 1
fi
source ${conf}

timestamp=`date +%Y-%m-%d-%H`
export OUT_DIR=${CDR_IE_ROOT}/saved_models/${MODEL_NAME}/${timestamp}/

additional_args=${@:2}


if [[ "$text_train" != "" ]]; then
    additional_args=" --text_train=$text_train $additional_args "
fi
if [[ "$kb_train" != "" ]]; then
    additional_args=" --kb_train=$kb_train $additional_args "
fi
if [[ "$positive_train" != "" ]]; then
    additional_args=" --positive_train=$positive_train $additional_args "
fi
if [[ "$negative_train" != "" ]]; then
    additional_args=" --negative_train=$negative_train $additional_args "
fi
if [[ "$positive_dist_train" != "" ]]; then
    additional_args=" --positive_dist_train=$positive_dist_train $additional_args "
fi
if [[ "$negative_dist_train" != "" ]]; then
    additional_args=" --negative_dist_train=$negative_dist_train $additional_args "
fi
if [[ "$positive_test" != "" ]]; then
    additional_args=" --positive_test=$positive_test $additional_args "
fi
if [[ "$negative_test" != "" ]]; then
    additional_args=" --negative_test=$negative_test $additional_args "
fi
if [[ "$positive_test_test" != "" ]]; then
    additional_args=" --positive_test_test=$positive_test_test $additional_args "
fi
if [[ "$negative_test_test" != "" ]]; then
    additional_args=" --negative_test_test=$negative_test_test $additional_args "
fi
if [[ "$all_ep_data" != "" ]]; then
    additional_args=" --all_ep_data=$all_ep_data $additional_args "
fi
if [[ "$pos_noise" != "" ]]; then
    additional_args=" --pos_noise=$pos_noise $additional_args "
fi
if [[ "$neg_noise" != "" ]]; then
    additional_args=" --neg_noise=$neg_noise $additional_args "
fi
if [[ "$type_file" != "" ]]; then
    additional_args=" --type_file=$type_file $additional_args "
fi
if [[ "$fb15k_dir" != "" ]]; then
    additional_args=" --fb15k_dir=$fb15k_dir $additional_args "
fi
if [[ "$nci_dir" != "" ]]; then
    additional_args=" --nci_dir=$nci_dir $additional_args "
fi
if [[ "$noise_dir" != "" ]]; then
    additional_args=" --noise_dir=$noise_dir $additional_args "
fi
if [[ "$candidate_file" != "" ]]; then
    additional_args=" --candidate_file=$candidate_file $additional_args "
fi
if [[ "$variance_file" != "" ]]; then
    additional_args=" --variance_file=$variance_file $additional_args "
fi
if [[ "$epsilon" != "" ]]; then
    additional_args=" --epsilon=$epsilon $additional_args "
fi
if [[ "$percentile" != "" ]]; then
    additional_args=" --percentile $additional_args "
fi
if [[ "$position_dim" != "" ]]; then
    additional_args=" --position_dim $position_dim $additional_args "
fi
if [[ "$text_encoder" != "" ]]; then
    additional_args=" --text_encoder $text_encoder $additional_args "
fi
if [[ "$kb_vocab_size" != "" ]]; then
    additional_args=" --kb_vocab_size $kb_vocab_size $additional_args "
fi
if [[ "$num_classes" != "" ]]; then
    additional_args=" --num_classes $num_classes $additional_args "
fi
if [[ "$variance_type" != "" ]]; then
    additional_args=" --variance_type $variance_type $additional_args "
fi
if [[ "$variance_min" != "" ]]; then
    additional_args=" --variance_min $variance_min $additional_args "
fi
if [[ "$variance_max" != "" ]]; then
    additional_args=" --variance_max $variance_max $additional_args "
fi
if [[ "$variance_delta" != "" ]]; then
    additional_args=" --variance_delta $variance_delta $additional_args "
fi
if [[ "$kb_batch" != "" ]]; then
    additional_args=" --kb_batch $kb_batch $additional_args "
fi
if [[ "$text_batch" != "" ]]; then
    additional_args=" --text_batch $text_batch $additional_args "
fi
if [[ "$ner_batch" != "" ]]; then
    additional_args=" --ner_batch $ner_batch $additional_args "
fi
if [[ "$eval_batch" != "" ]]; then
    additional_args=" --eval_batch $eval_batch $additional_args "
fi
if [[ "$load_batch" != "" ]]; then
    additional_args=" --load_batch $load_batch $additional_args "
fi
if [[ "$kb_pretrain" != "" ]]; then
    additional_args=" --kb_pretrain $kb_pretrain $additional_args "
fi
if [[ "$log_every" != "" ]]; then
    additional_args=" --log_every $log_every $additional_args "
fi
if [[ "$max_steps" != "" ]]; then
    additional_args=" --max_steps $max_steps $additional_args "
fi
if [[ "$null_label" != "" ]]; then
    additional_args=" --null_label $null_label $additional_args "
fi
if [[ "$beta2" != "" ]]; then
    additional_args=" --beta2 $beta2 $additional_args "
fi
if [[ "$beta1" != "" ]]; then
    additional_args=" --beta1 $beta1 $additional_args "
fi
if [[ "$word_unk_dropout" != "" ]]; then
    additional_args=" --word_unk_dropout $word_unk_dropout $additional_args "
fi
if [[ "$pos_unk_dropout" != "" ]]; then
    additional_args=" --pos_unk_dropout $pos_unk_dropout $additional_args "
fi
if [[ "$dropout_loss_weight" != "" ]]; then
    additional_args=" --dropout_loss_weight $dropout_loss_weight $additional_args "
fi
if [[ "$kg_label_file" != "" ]]; then
    additional_args=" --kg_label_file $kg_label_file $additional_args "
fi
if [[ "$ner_train" != "" ]]; then
    additional_args=" --ner_train $ner_train $additional_args "
fi
if [[ "$ner_test" != "" ]]; then
    additional_args=" --ner_test $ner_test $additional_args "
fi
if [[ "$ner_weight" != "" ]]; then
    additional_args=" --ner_weight $ner_weight $additional_args "
fi
if [[ "$ner_prob" != "" ]]; then
    additional_args=" --ner_prob $ner_prob $additional_args "
fi
if [[ "$embeddings" != "" ]]; then
    additional_args=" --embeddings $embeddings $additional_args "
fi
if [[ "$max_decrease_epochs" != "" ]]; then
    additional_args=" --max_decrease_epochs $max_decrease_epochs $additional_args "
fi
if [[ "$block_repeats" != "" ]]; then
    additional_args=" --block_repeats $block_repeats $additional_args "
fi
if [[ "$pos_prob" != "" ]]; then
    additional_args=" --pos_prob $pos_prob $additional_args "
fi
if [[ "$layer_str" != "" ]]; then
    additional_args=" --layer_str $layer_str $additional_args "
fi
if [[ "$noise_std" != "" ]]; then
    additional_args=" --noise_std $noise_std $additional_args "
fi
if [[ "$doc_filter" != "" ]]; then
    additional_args=" --doc_filter $doc_filter $additional_args "
fi
if [[ "$train_dev_percent" != "" ]]; then
    additional_args=" --train_dev_percent $train_dev_percent $additional_args "
fi
if [[ "$label_weights" != "" ]]; then
    additional_args=" --label_weights $label_weights $additional_args "
fi
if [[ "$thresholds" != "" ]]; then
    additional_args=" --thresholds $thresholds $additional_args "
fi


# Boolean args
if [[ "$norm_entities" != "" ]]; then
    additional_args=" --norm_entities $additional_args "
fi
if [[ "$bidirectional" != "" ]]; then
    additional_args=" --bidirectional $additional_args "
fi
if [[ "$use_tanh" != "" ]]; then
    additional_args=" --use_tanh $additional_args "
fi
if [[ "$use_peephole" != "" ]]; then
    additional_args=" --use_peephole $additional_args "
fi
if [[ "$max_pool" != "" ]]; then
    additional_args=" --max_pool $additional_args "
fi
if [[ "$center_only" != "" ]]; then
    additional_args=" --center_only $additional_args "
fi
if [[ "$arg_entities" != "" ]]; then
    additional_args=" --arg_entities $additional_args "
fi
if [[ "$mlp" != "" ]]; then
    additional_args=" --mlp $additional_args "
fi
if [[ "$reduce_max" != "" ]]; then
    additional_args=" --reduce_max $additional_args "
fi
if [[ "$in_memory" != "" ]]; then
    additional_args=" --in_memory $additional_args "
fi



export CMD="python $train \
--vocab_dir=$vocab_dir \
--optimizer=$optimizer \
--loss_type=$loss_type \
--model_type=$model_type \
--lr=$lr \
--margin=$margin \
--l2_weight=$l2_weight \
--word_dropout=$word_dropout \
--lstm_dropout=$lstm_dropout \
--final_dropout=$final_dropout \
--clip_norm=$clip_norm \
--text_weight=$text_weight \
--text_prob=$text_prob \
--token_dim=$token_dim \
--lstm_dim=$lstm_dim \
--embed_dim=$embed_dim \
--kb_epochs=$kb_epochs \
--text_epochs=$text_epochs \
--eval_every=$eval_every \
--max_seq=$max_seq \
--neg_samples=$neg_samples \
--random_seed=$random_seed \
$additional_args"
