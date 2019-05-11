python ./data_processing/data_preparation.py \
--data_path=./data_features/CHB-MIT_features \
--patient=2 \
--preictal_duration=2700 \
--discard_data=True \
--discard_data_duration=180 \
--features_names max_correlation DSTL SPLV nonlinear_interdependence