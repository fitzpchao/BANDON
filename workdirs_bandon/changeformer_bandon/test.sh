#!/usr/bin/env bash
set -x

 CONFIG_ROOT=$(dirname "$0")
 TRAIN_CONFIG=config.py
 JOB_NAME=mmcd_test
 CONFIG_ROOT_eval=./configs/bandon
 GPU=8

# -------------------------------------------------------------------
 WORK_DIR=./$(dirname "$0")/${TRAIN_CONFIG%.*}_GPU${GPU}
 echo $WORK_DIR
mkdir -p ${WORK_DIR}

 CHECKPOINT=${WORK_DIR}/$1
# -------------------------------------------------------------------


for TEST_DATASET in \
    BANDON_test \
    BANDON_test_ood
do
    TEST_DATA_CONFIG=${CONFIG_ROOT_eval}/testdata/${TEST_DATASET}.py
    OUT_DIR=${CHECKPOINT%.*}/${TEST_DATASET}
    echo $OUT_DIR
    mkdir -p ${OUT_DIR}

    PORT=$((12000 + $RANDOM % 20000))
    # -------------------------------------------------------------------

    now=$(date +"%Y%m%d_%H%M%S")
    
    PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
    python -u \
        tools/test_rs_mtl_index.py \
        $CONFIG_ROOT/$TRAIN_CONFIG \
        $CHECKPOINT \
        $TEST_DATA_CONFIG \
        --out_dir=${OUT_DIR} \
        2>&1 | tee ${OUT_DIR}/test-$now.log
 done

#CPUS_PER_TASK=${CPUS_PER_TASK:-5}
#        --cpus-per-task=${CPUS_PER_TASK} \
