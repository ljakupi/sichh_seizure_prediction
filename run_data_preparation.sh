#!/usr/bin/env bash
python ./data_processing/data_preparation.py \
--data_path=./data_features/CHB-MIT_features/sec_30 \
--patient=12 \
--preictal_duration=900 \
--discard_data=True \
--discard_data_duration=60 \
--features_names max_correlation max_correlation_unbiased DSTL SPLV univariate