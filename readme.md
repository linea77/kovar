<div align="center">    
 
# Korean visual abductive reasoning: AI Language Model’s ability to understand plausibility
 

[![Paper](https://img.shields.io/badge/paper-Linguistic_Research_v41--2-red)](http://isli.khu.ac.kr/journal/content/data/41_2/4.pdf)
[![Conference](https://img.shields.io/badge/ALAK-2024-blue)]()

</div>
 
## Description   
This repository is to archive the official implementation and experimental results for the paper "Korean visual abductive reasoning: AI Language Model's ability to understand plausibility" by Seonah Han, Jongbin Won, Eunjae Kwon, and Sanghoun Song.

## Key Contributions
This study investigates how a multimodal language model numerically estimates the plausibility of Korean hypothesis sentences in visual abductive reasoning tasks. The research utilizes a dual encoder model and the Korean Story Cloze dataset to analyze the model's ability to understand and compare the plausibility of different hypotheses.


## How to run   
### Train
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

### Inference
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


### Citation   
```
@article{seonah2024korean,
  title={Korean visual abductive reasoning: AI Language Model’s ability to understand plausibility},
  author={Seonah, Han and Jongbin, Won and Eunjae, Kwon and Sanghoun, Song},
  journal={Linguistic Research},
  year={2024},
  volume={41},
  number={2},
  pages={283--310}
}
```   


### LICENSE
MIT