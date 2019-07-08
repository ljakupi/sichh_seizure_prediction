#!/usr/bin/env bash
python ./train_eval_model.py \
--data_path=./processed_datasets/CHB-MIT \
--patient=13 \
--model=TCN \
--preictal_duration=1800 \
--discard_data_duration=60 \
--group_segments_form_input=True \
--n_segments_form_input=5 \
--segmentation_duration=30