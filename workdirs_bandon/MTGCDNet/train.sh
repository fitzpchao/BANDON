#!/usr/bin/env sh

# -------------------------------------------------------------------
 CONFIG_ROOT=$(dirname "$0")
#TRAIN_CONFIG=t2.6.3.2.py
#TRAIN_CONFIG=A2.1.py
 TRAIN_CONFIG=config.py
 JOB_NAME=${TRAIN_CONFIG}
 GPU=$1


 WORK_DIR=./$(dirname "$0")/${TRAIN_CONFIG%.*}_GPU${GPU}
 echo $WORK_DIR
# -------------------------------------------------------------------
mkdir -p ${WORK_DIR}
now=$(date +"%Y%m%d_%H%M%S")

./tools/dist_train.sh $CONFIG_ROOT/$TRAIN_CONFIG $GPU \
--work-dir=${WORK_DIR} \
--no-validate \
2>&1 | tee ${WORK_DIR}/train-$now.log

