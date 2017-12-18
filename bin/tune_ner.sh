#!/usr/bin/env bash

config=$1
additional_args=${@:2}
add_arg_str=`echo ${additional_args} | tr ' ' '_' | tr '/' '-'`

MEM=25GB
num_gpus=32
gpu_partition=titanx-short


lrs=".001 .0005"
kb_batch_sizes="128"
text_batch_sizes="32"
ner_weights="1.0"
text_weights="1.0"
text_probs="1.0"
l2s="0"
drop_losses="0"
margins="1.0"
embed_dims="64 256 512"
lstm_dims="0"
token_dims="64 128"
position_dims="0"
word_unk_dropouts=".45 .65"
pos_unk_dropouts="1.0"
word_dropouts=".45 .65"
lstm_dropouts=".65 .85"
final_dropouts=".65 .85"
epsilons="1e-8"
variance_mins="0"
beta_2s=".9"
block_repeats="1"
repeats="1"

# generate basic run cmd
source ${CDR_IE_ROOT}/bin/gen_run_cmd.sh ${config} ${additional_args}

for _lr in ${lrs[@]}; do
    for _kb_batch in ${kb_batch_sizes[@]}; do
        for _text_batch in ${text_batch_sizes[@]}; do
            for _ner_weight in ${ner_weights[@]}; do
                for _text_weight in ${text_weights[@]}; do
                    for _text_prob in ${text_probs[@]}; do
                        for _l2 in ${l2s[@]}; do
                            for _drop_loss in ${drop_losses[@]}; do
                                for _margin in ${margins[@]}; do
                                    for _embed_dim in ${embed_dims[@]}; do
                                        for _token_dim in ${token_dims[@]}; do
                                            for _lstm_dim in ${lstm_dims[@]}; do
                                                for _position_dim in ${position_dims[@]}; do
                                                    for _pos_unk_dropout in ${pos_unk_dropouts[@]}; do
                                                        for _word_unk_dropout in ${word_unk_dropouts[@]}; do
                                                            for _word_dropout in ${word_dropouts[@]}; do
                                                                for _lstm_dropout in ${lstm_dropouts[@]}; do
                                                                    for _final_dropout in ${final_dropouts[@]}; do
                                                                        for _epsilon in ${epsilons[@]}; do
                                                                            for _var_min in ${variance_mins[@]}; do
                                                                                for _beta2 in ${beta_2s[@]}; do
                                                                                    for _block_repeat in ${block_repeats[@]}; do
                                                                                        for _repeat in ${repeats[@]}; do
                                                                                        # hack to not run double non-variance jobs
                                            #                                            if [[ "$_var_type" == "divide" ]] || [[ "$_var_weight" != "0.0" ]];
                                            #                                            then
                                                                                            OUT_LOG="${OUT_DIR}/${_lr}_${_kb_batch}_${_text_batch}_${_ner_weight}_${_text_weight}_${_text_prob}_${_l2}_${_drop_loss}_${_margin}_${_embed_dim}_${_lstm_dim}_${_token_dim}_${_position_dim}_${_pos_unk_dropout}_${_word_unk_dropout}_${_word_dropout}_${_lstm_dropout}_${_final_dropout}_${_epsilon}_${_beta2}_${_var_min}_${_block_repeat}_${_repeat}_${add_arg_str}"
                                                                                            mkdir -p ${OUT_LOG}
                                                                                            tune_args="\
                                                                                            --logdir ${OUT_LOG} \
                                                                                            --random_seed ${_repeat} \
                                                                                            --lr $_lr \
                                                                                            --margin $_margin \
                                                                                            --text_batch $_text_batch \
                                                                                            --ner_weight $_ner_weight \
                                                                                            --text_weight $_text_weight \
                                                                                            --text_prob $_text_prob \
                                                                                            --kb_batch $_kb_batch \
                                                                                            --l2_weight $_l2 \
                                                                                            --dropout_loss_weight $_drop_loss \
                                                                                            --embed_dim $_embed_dim \
                                                                                            --lstm_dim $_embed_dim \
                                                                                            --word_unk_dropout $_word_unk_dropout \
                                                                                            --pos_unk_dropout $_pos_unk_dropout \
                                                                                            --word_dropout $_word_dropout \
                                                                                            --lstm_dropout $_lstm_dropout \
                                                                                            --final_dropout $_final_dropout \
                                                                                            --token_dim $_token_dim \
                                                                                            --position_dim $_position_dim \
                                                                                            --variance_min $_var_min \
                                                                                            --block_repeats $_block_repeat \
                                                                                            --beta2 $_beta2 \
                                                                                            --epsilon $_epsilon "
                                                                                            commands+=("srun --gres=gpu:1 --mem=$MEM --partition=${gpu_partition} ${CMD} $tune_args \
                                                                                                &> $OUT_LOG/train.log")
                                                                                            echo ${CMD} ${tune_args} > ${OUT_LOG}/run.cmd
                                            #                                            fi
                                                                                        done
                                                                                    done
                                                                                done
                                                                            done
                                                                        done
                                                                    done
                                                                done
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# now distribute them to the gpus
num_jobs=${#commands[@]}
jobs_per_gpu=$((num_jobs / num_gpus))
echo "Distributing $num_jobs jobs to $num_gpus gpus ($jobs_per_gpu jobs/gpu)"

j=0
for (( gpuid=0; gpuid<num_gpus; gpuid++)); do
    for (( i=0; i<jobs_per_gpu; i++ )); do
        jobid=$((j * jobs_per_gpu + i))
        comm="${commands[$jobid]}"
        comm=${comm/XX/$gpuid}
        echo "Starting job $jobid on gpu $gpuid"
#        echo ${comm}
        eval ${comm}
    done &
    j=$((j + 1))
done

