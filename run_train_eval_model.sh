python ./train_eval_model.py \
--data_path=./processed_datasets/CHB-MIT \
--patient=2 \
--model=RNN \
--preictal_duration=1800 \
--group_segments_form_input=True \
--n_segments_form_input=10