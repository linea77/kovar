import os
import csv
from typing import *
import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

from kovar.model import DualEncoderModelForMultipleChoice
from kovar.preprocess import KoVARDataset, KoVARDataCollator

def make_parser():
    parser  = argparse.ArgumentParser()

    parser.add_argument("--test_path",
                        type=str,
                        help='path of a test set'
                        )
    parser.add_argument("--image_path",
                        type=str)
    parser.add_argument("--model_path",
                        type=str,
                        default=None,
                        help='path of the model checkpoint')
    parser.add_argument("--output_dir",
                        type=str)

    parser.add_argument("--text_format",
                       type = str,
                       default=None)
    parser.add_argument("--black_image",
                       action = "store_true")
    
    parser.add_argument("--device",
                        type=str,
                        default='cpu')
    parser.add_argument("--batch_size",
                        type=int,
                        default=4)

    return parser
args = make_parser().parse_args()
print(args)

def show_model_state_info(checkpoint):
    print("**Model State Info**")
    print(f'best epoch : {checkpoint["epoch"]}') 
    print(f'best step : {checkpoint["step"]}') 
    print(f'best valid_loss : {checkpoint["valid_loss"]}') 
    print(f'best valid_acc : {checkpoint["valid_acc"]}') 
    print(f'best train_loss : {checkpoint["train_loss"]}') 


def main():
    sentence_types =['GROUNDTRUTH', 'PLAUSIBLE', 'IMPLAUSIBLE', 'RANDOM']

    # Load dataset & data loader
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    image_processor = AutoImageProcessor.from_pretrained('google/vit-large-patch16-224')
    
    test_set = KoVARDataset(dataset_path = args.test_path, 
                            image_path = args.image_path,
                            all_columns=True)
    

    collator = KoVARDataCollator(image_processor = image_processor,
                                tokenizer = tokenizer,
                                text_format=args.text_format,
                                use_story_id=True,
                                sentence_types=sentence_types)
    
    test_loader = DataLoader(test_set,
                              batch_size = args.batch_size,
                              shuffle=False,
                              collate_fn = collator
                              )

    
    # Load model
    text_encoder = AutoModel.from_pretrained('klue/roberta-large')
    image_encoder = AutoModel.from_pretrained('google/vit-large-patch16-224')    
    model = DualEncoderModelForMultipleChoice(
        image_encoder = image_encoder,
        text_encoder=text_encoder, 
                       )   
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        show_model_state_info(checkpoint)
    model.to(args.device)
    model.eval()

    # Saving Directory & File Name
    save_dir = os.path.join(args.output_dir, 'inference_result')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.model_path:
        fname = args.model_path.split('/')[-1].split('.')[0]+'_inference_4_hyps.csv'
    else:
        fname = 'inference_4_hyps.csv' # annotation artifact
       
    save_path = os.path.join(save_dir, fname)
   
    with open(save_path, "w", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['IDX', 'STORY_ID', 'SentenceType', 'Logit', 'Softmax'])
    

    index = 0
    test_pbar = tqdm(test_loader,
                     total=len(test_loader),
                     desc=f'TEST')
    with torch.no_grad():
        for batch in test_pbar:
            inputs, _, story_ids = batch

            for key, value in inputs.items():
                inputs[key] = value.to(args.device)
            
            output = model(**inputs )
            logits = output.logits
            probs = torch.softmax(logits, dim=1)

            batch_size, num_hyps = logits.size()
            
            for i in range(batch_size):
                STORY_ID = story_ids[i]

                for j in range(num_hyps):
                    index += 1
                    IDX = index
                    SENTENCE_TYPE = sentence_types[j]
                    LOGIT = logits[i][j].item()
                    SOFTMAX = probs[i][j].item()

                    with open(save_path, 'a', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([IDX, STORY_ID, SENTENCE_TYPE, LOGIT, SOFTMAX])


if __name__ == '__main__':
    main()