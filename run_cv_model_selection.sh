#!/usr/bin/env bash
python ./cv_model_selection.py \
--data_path=./processed_datasets/CHB-MIT \
--patient=6 \
--model=TCN \
--preictal_duration=1800 \
--discard_data_duration=60 \
--group_segments_form_input=True \
--n_segments_form_input=10 \
--segmentation_duration=30 \
--CV=5