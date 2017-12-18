#!/usr/bin/env bash

config=$1
ARGS=${@:2}

MEM=100GB
PARTITION=m40-short

source ${CDR_IE_ROOT}/bin/gen_run_cmd.sh $config $ARGS

OUT_LOG="${OUT_DIR}/${RANDOM}_${RANDOM}"
echo "Writing to $OUT_LOG"

CMD="srun --gres=gpu:1 --partition=$PARTITION --mem=$MEM ${CMD} --logdir=$OUT_LOG"
echo ${CMD}
mkdir -p ${OUT_LOG}
echo ${CMD} > ${OUT_LOG}/run.cmd

eval "${CMD} 2>&1 | tee ${OUT_LOG}/train.log"
