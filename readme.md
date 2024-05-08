# Train
```
python3 train.py \
--project={your_wandb_project_name} \
--logging_steps={wandb_logging_steps} \
--seed=42 \
--freeze_text_encoder \
--freeze_image_encoder \
--batch_size=32 \
--num_epoch=30 \
--early_stop=25 \
--lr=5e-5 \
--device=cuda \
--output_dir='./output' \
--train_path={your/path/to/train_set.json} \
--valid_path={your/path/to/valid_set.json} \
--image_path={your/path/to/image_dir/} \
--text_format='{obs1} {hyp} {obs2}'
```

# Inference
* 4 hypotheses
```
python3 inference_4_hyps.py \
--test_path={your/path/to/test_set.json} \
--image_path={your/path/to/image_dir/} \
--model_path='./output/model/{model_checkpoint.pth}' \
--output_dir='./output' \
--text_format='{obs1} {hyp} {obs2}' \
--device=cpu \
--batch_size=32
```

* 2 hypothesis
```
python3 inference_2_hyps.py \
--test_path={your/path/to/test_set.json} \
--image_path={your/path/to/image_dir/} \
--model_path='./output/model/{model_checkpoint.pth}' \
--output_dir='./output' \
--text_format='{obs1} {hyp} {obs2}' \
--device=cpu \
--batch_size=32
```