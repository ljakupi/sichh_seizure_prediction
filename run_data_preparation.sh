#!/usr/bin/env bash
python ./data_processing/data_preparation.py \
--data_path=./data_features/CHB-MIT_features \
--patient=3 \
--preictal_duration=3600 \
--discard_data=True \
--discard_data_duration=60 \
--features_names max_correlation DSTL SPLV nonlinear_interdependence univariate