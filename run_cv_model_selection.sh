#!/usr/bin/env bash
python ./cv_model_selection.py \
--data_path=./processed_datasets/CHB-MIT \
--patient=3 \
--model=FC \
--preictal_duration=1800 \
--discard_data_duration=60 \
--group_segments_form_input=False \
--n_segments_form_input=10 \
--CV=5