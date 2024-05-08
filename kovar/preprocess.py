import os

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from datasets import Dataset

class KoVARDataset:
    def __init__(self, 
                 dataset_path,
                 image_path,
                #  image_type:str=None,
                 all_columns=False
                  ):
        self.image_path = image_path
        df = pd.read_json(dataset_path, lines=True)

        df['image_path'] = df['CLUE1'].map(self._get_image_paths)
    
        # if image_type =='random':
        #    print('Use Random Image')
        #    df['image_path'] = df['image_path'].sample(frac=1).reset_index(drop=True)

        if not all_columns:
            df = df.loc[:,['OBS1', 'hyp0', 'hyp1', 'OBS2', 'label', 'image_path']]
        
        self.dataset = Dataset.from_pandas(df)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def _get_image_paths(self, image_id:str) -> str:
        return os.path.join(self.image_path, image_id[:3], f"{image_id}.jpg")


class KoVARDataCollator:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """
    def __init__(self, 
                 vl_processor=None,
                 image_processor=None,
                 tokenizer=None, 
                 text_format=None,
                 use_black_image=False,
                 use_story_id = False,
                 sentence_types = ['hyp0', 'hyp1']
                 ):
        
        if (vl_processor==None) :
            self.vl_processor=None
            if (tokenizer==None) and (image_processor==None):
                raise ValueError("Either `vl_processor` or `tokenizer` is needed.")
            else:
                self.image_processor = image_processor 
                self.tokenizer = tokenizer 
        else:
            self.vl_processor = vl_processor
            self.tokenizer=None
            self.image_processor=None

        self.text_format = text_format
        self.use_black_image = use_black_image
        self.use_story_id = use_story_id
        self.sentence_types=sentence_types

    def __call__(self, examples):
        if self.use_story_id:
            story_ids =  [example.pop('STORY_ID')for example in examples] 
        else:
            story_ids = None

        labels = [example.pop('label')for example in examples]
        max_length = 0

        batch  = {'input_ids':[],
                  'attention_mask':[],
                  'token_type_ids':[],
                  'pixel_values':[]}
        
        for example in examples:
            obs1 = example.pop('OBS1')
            hyps = [example.pop(sentence_type) for sentence_type in self.sentence_types]
            obs2 = example.pop('OBS2')

            if self.use_black_image:
                image = np.zeros(shape=(1000, 1000, 3))
            else:
                image = Image.open(example.pop('image_path'))
                image = ImageOps.exif_transpose(image)

            input_text= [self.text_format.format(obs1=obs1, hyp=hyp, obs2=obs2) for hyp in hyps]

            if self.vl_processor:
                processed_input = self.vl_processor(
                    images = image,
                    text = input_text,
                    return_tensors  = 'pt',
                    padding=True
                )
            else :
                processed_input = {}
                processed_input.update(self.image_processor(image, return_tensors='pt'))
                processed_input.update(self.tokenizer(input_text, return_tensors='pt', padding=True))
    

            if processed_input['input_ids'].size(-1) > max_length:
                max_length = processed_input['input_ids'].size(-1)
            
            for key, value in processed_input.items():
                if key == 'pixel_values':
                    repeated_pixel_values = [value] * len(self.sentence_types)
                    batch[key].append(torch.cat(repeated_pixel_values, dim=0))
                else:
                    batch[key].append(value)
                

        inputs = self._dynamic_padding(batch, batch_size=len(examples), max_length=max_length)
        inputs['pixel_values'] = torch.stack(inputs['pixel_values'], dim=0)

        if self.use_story_id:
            return inputs, labels, story_ids
        else:
            return inputs, labels

    def _dynamic_padding(self, input_batch:dict, 
                         batch_size:int,
                         max_length:int):
        if self.vl_processor : 
            pad_token_id = self.vl_processor.tokenizer.pad_token_id
        else :
            pad_token_id = self.tokenizer.pad_token_id

        def add_pad_token_id(x):
            num_pad = (max_length - x.shape[-1])
            return np.pad(x, ((0,0), (0, num_pad)), 'constant', constant_values=pad_token_id)
        
        def add_zero(x):
            num_pad = (max_length - x.shape[-1])
            return np.pad(x, ((0,0), (0, num_pad)), 'constant', constant_values=0)

        input_ids = list(map(add_pad_token_id, input_batch['input_ids'] ))
        token_type_ids = list(map(add_zero, input_batch['token_type_ids']))
        attention_mask = list(map(add_zero, input_batch['attention_mask']))

        input_batch['input_ids'] = torch.tensor(np.array(input_ids)).reshape(batch_size, -1, max_length)
        input_batch['token_type_ids'] = torch.tensor(np.array(token_type_ids)).reshape(batch_size, -1, max_length) 
        input_batch['attention_mask'] = torch.tensor(np.array(attention_mask)).reshape(batch_size, -1, max_length)

        return input_batch
    