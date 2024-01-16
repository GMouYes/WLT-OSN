#!/bin/bash
EXPNAME="vit_flat_img_text_ocr_lr1e_epoch8_hidden512"

echo "EXPNAME: ${EXPNAME}"
CUDA_VISIBLE_DEVICES=0 nohup python3 -u -W ignore trainer.py \
  --exp_name ${EXPNAME}"/" \
  --configReadPath "../../config/${EXPNAME}.yml" \
  --do_grid_search \
  > ../../logs/${EXPNAME}_train.out &


