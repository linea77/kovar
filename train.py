import os
import random
import argparse
from datetime import datetime

import wandb 
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

from kovar.preprocess import KoVARDataset, KoVARDataCollator
from kovar.model import DualEncoderModelForMultipleChoice

def make_parser():
    parser  = argparse.ArgumentParser()
    parser.add_argument("--project",
                        type=str,
                        help='wandb; project name',
                        default="kovar")
    parser.add_argument("--logging_steps",
                        type=int,
                        help='wandb; `train_loss` logging step',
                        default=5)
    
    parser.add_argument("--seed",
                    type=int,
                    default=42)
    parser.add_argument("--freeze_text_encoder",
                        action = "store_true")
    parser.add_argument("--freeze_image_encoder",
                        action = "store_true")
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=4)
    parser.add_argument("--num_epoch",
                        type=int,
                        default=30)
    parser.add_argument("--early_stop",
                        type=int,
                        default=25)
    parser.add_argument("--lr",
                        type=float,
                        default=5e-5)
    
    parser.add_argument("--device",
                        type=str,
                        default='cuda:0')

    parser.add_argument("--output_dir",
                        type=str)
    parser.add_argument("--train_path",
                        type=str)
    parser.add_argument("--valid_path",
                        type=str)
    parser.add_argument("--image_path",
                        type=str)

    
    parser.add_argument("--black_image",
                       action = "store_true")
 
    parser.add_argument("--text_format",
                       type = str,
                       default=None)
    
    
    parser.add_argument("--last_epoch",
                        type=int,
                        default=None)
    parser.add_argument("--last_step",
                        type=int,
                        default=None)
    parser.add_argument("--last_checkpoint",
                        type=int,
                        default=None)
    return parser
args = make_parser().parse_args()
print(args)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def main():
    run = wandb.init(project = args.project )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%s")

    # dataset 
    train_set = KoVARDataset(dataset_path = args.train_path, 
                                       image_path = args.image_path,
                                       all_columns =False)
    valid_set = KoVARDataset(dataset_path = args.valid_path, 
                                       image_path = args.image_path,
                                       all_columns =False)

    # data collator & data loader
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    collator = KoVARDataCollator(image_processor = image_processor,
                                 tokenizer = tokenizer,
                                 text_format=args.text_format,
                                 use_black_image=args.black_image)
    
    train_loader = DataLoader(train_set,
                              batch_size = args.batch_size,
                              shuffle=False,
                              collate_fn = collator
                              )
    valid_loader = DataLoader(valid_set,
                              batch_size = args.batch_size,
                              shuffle=False,
                              collate_fn = collator
                              )

    
    # model
    text_encoder = AutoModel.from_pretrained('klue/roberta-large')
    image_encoder = AutoModel.from_pretrained('google/vit-large-patch16-224')
    model = DualEncoderModelForMultipleChoice(image_encoder = image_encoder,
                       text_encoder=text_encoder, 
                       freeze_text_encoder=args.freeze_text_encoder,
                       freeze_image_encoder=args.freeze_image_encoder,
                       )   
    model.to(args.device)
    
    wandb.watch(model, log='all')
    print(f'{wandb.run.id} {wandb.run.name} {timestamp}')
    print(args)
       
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.lr)

    model_save_path = os.path.join(args.output_dir, 'model')
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    
    best_valid_loss= 1_000_000.0
    running_loss = 0.
    early_stop = 0

    step = -1
    last_step = -1 
    start = args.last_epoch if args.last_epoch else 0
    end = (start + args.num_epoch) 
       
    print("**training loop**")
    for epoch in range(start, (end)):
        if early_stop >= args.early_stop:
            print("**Early Stop**")
            break

        train_pbar = tqdm(train_loader,
                          total=len(train_loader),
                          desc = f'EPOCH{epoch+1}',
                          leave = False)
        for batch in train_pbar:
            step += 1
            if step < last_step:
                continue

            torch.cuda.empty_cache()
            optimizer.zero_grad()
            model.train()

            inputs, labels = batch

            for key, value in inputs.items():
                inputs[key] = value.to(args.device)
            
            output = model(labels = labels, **inputs)
            loss = output.loss
            loss.backward()
            optimizer.step()
            # scheduler.step()

            running_loss += loss.item()
            train_pbar.set_postfix({'train_loss':loss.item()})

            if (step % args.logging_steps == 0):
                last_loss = (running_loss / args.logging_steps)
                run.log({'train_loss':last_loss}, step = step)
                running_loss = 0.
            
        # validation
        model.eval()
        valid_loss = 0.
        valid_acc = 0.
        eval_pbar = tqdm(valid_loader,
                         total=len(valid_loader),
                         desc = f'Epoch{epoch+1} VALID',
                         leave=False)
                
        with torch.no_grad():
            for batch in eval_pbar:
                inputs, labels = batch
                for key, value in inputs.items():
                    inputs[key] = value.to(args.device)
                
                output=model(labels=labels, **inputs)
                
                loss = output.loss
                probs = torch.softmax(output.logits, dim=1)
                _, pred = torch.max(probs, 1)
                
                acc = (pred == output.labels).float().mean().item()
                
                valid_loss += loss.item()
                valid_acc += acc
                eval_pbar.set_postfix({'valid_loss':loss.item(),
                                       'valid_acc':acc})
            avg_valid_loss = (valid_loss/len(eval_pbar))
            avg_valid_acc = (valid_acc/len(eval_pbar))
            run.log({'valid_loss':avg_valid_loss,
                     'valid_acc':avg_valid_acc*100}, step = step)
            print(avg_valid_acc)
                
        # save model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            model_path = os.path.join(model_save_path, f'best_{timestamp}_{wandb.run.project}_{wandb.run.name}.pth')
            save_dict = {
                'epoch':epoch,
                'step':step,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'train_loss':last_loss,
                'valid_loss':avg_valid_loss,
                'valid_acc':avg_valid_acc
            }
            torch.save(save_dict, model_path)
        else:
            early_stop += 1
            print('Early Stop Count: ', early_stop)


if __name__ == '__main__':
    seed_everything(args.seed)
    main()