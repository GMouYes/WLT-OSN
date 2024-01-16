#!/bin/bash
EXPNAME="vit_flat_img_text_ocr_lr1e-5_epoch8_hidden512"

echo "EXPNAME: ${EXPNAME}"
CUDA_VISIBLE_DEVICES=0 nohup python3 -u -W ignore validator.py \
    --exp_name ${EXPNAME}"/" \
    --configReadPath "../../output/${EXPNAME}/config.yml" \
    --newOutputPath "../../output/${EXPNAME}/validate" \
    --dataPath  "../../data/" \
    --textPath  ".csv" \
    --imagePath "../../data/images/" \
    > ../../logs/${EXPNAME}_valid.out &