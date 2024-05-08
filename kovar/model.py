
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

@dataclass
class ModelOutput():
    logits:torch.tensor = None
    loss:torch.tensor =None

class DualEncoderModelForMultipleChoice(nn.Module):
    def __init__(self, 
                 text_encoder, 
                 image_encoder,
                 freeze_text_encoder:bool=True,
                 freeze_image_encoder:bool=True):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
            
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.image_projection = nn.Linear(1024, 512)
        self.text_projection  = nn.Linear(1024, 512)
        self.dense = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(1024, 1)

        
    def forward(self,
                pixel_values,
                input_ids, 
                token_type_ids, 
                attention_mask, 
                labels=None):
        
        # (batch, num_choices, length)
        batch_size, num_choices , _= input_ids.size() 

        # (batch*num_choices, 3, 224, 224 )
        flat_pixel_values = pixel_values.reshape(num_choices*batch_size, 3,224,224)
        
        # (batch*num_choices, length)
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) 
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) 
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))  

        # Outputs of each encoder
        vision_outputs = self.image_encoder(pixel_values = flat_pixel_values)
        text_outputs = self.text_encoder(input_ids=flat_input_ids, 
                                    token_type_ids = flat_token_type_ids, 
                                    attention_mask=flat_attention_mask) 
        
        # (batch*num_choices, 768)
        image_features = vision_outputs['last_hidden_state'][:,0,:]
    
        # (batch*num_choices, 1024)
        text_features = text_outputs['last_hidden_state'][:,0,:]
        
        # (batch*num_choices, 512)
        image_features = self.image_projection(image_features)  
        text_features = self.text_projection(text_features)     

        # (batch, 1024)
        x = torch.cat([image_features, text_features], dim=1)  
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        # logits shape (batch, 1)
        logits = self.classifier(x)

        # reshaped logits (batch * num_choices)
        reshaped_logits = logits.view(-1, num_choices)
            
        if labels:
            labels = torch.tensor(np.array(labels), dtype=torch.long, device=logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        else:
            loss=None

        return ModelOutput(
            logits = reshaped_logits,
            loss = loss
        )
    