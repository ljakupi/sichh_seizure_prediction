python ./cv_model_selection.py \
--data_path=./processed_datasets/CHB-MIT \
--patient=2 \
--model=RNN \
--CV=3 \
--preictal_duration=1800 \
--group_segments_form_input=True \
--n_segments_form_input=10