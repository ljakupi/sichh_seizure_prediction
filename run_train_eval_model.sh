#!/usr/bin/env bash
python ./train_eval_model.py \
--data_path=./processed_datasets/CHB-MIT \
--patient=12 \
--model=FC \
--preictal_duration=900 \
--discard_data_duration=60 \
--group_segments_form_input=False \
--n_segments_form_input=5 \
--segmentation_duration=30