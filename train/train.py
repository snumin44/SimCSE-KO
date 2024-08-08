import os
import re
import time
import random
import datetime
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast 
from torch.cuda.amp import GradScaler
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

import sys
sys.path.append("../") 

from src.data_loader import Dataset_CSV, Dataset_TXT, Dataset_STS, DataCollator
from src.model import SimCSE
from evaluation.utils import inference 

LOGGER = logging.getLogger()

def argument_parser():

    parser = argparse.ArgumentParser(description='train simcse')

    # Required
    parser.add_argument('--model', type=str, required=True,
                        help='Directory for pretrained model'
                       )    
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Directory for pretrained tokenizer'
                       )     
    parser.add_argument('--train_data', type=str, required=True,
                        help='Training set directory'
                       )
    parser.add_argument('--valid_data', type=str, default='kor_dev',
                        help='Validation set : {kor_dev|klue_dev}'
                       )
    parser.add_argument('--output_path', type=str, default='../output/checkpoint',
                        help='Directory for output'
                       )
    
    # Tokenizer & Collator settings
    parser.add_argument('--max_length', default=64, type=int,
                        help='Max length of sequence'
                       )
    parser.add_argument('--padding', action="store_false", default=True,
                        help='Add padding to short sentences'
                       )
    parser.add_argument('--truncation', action="store_false", default=True,
                        help='Truncate extra tokens when exceeding the max_length'
                       )
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size'
                       )
    parser.add_argument('--shuffle', action="store_false", default=True,
                        help='Load shuffled sequences'
                       )

    # Train config    
    parser.add_argument('--epochs', default=1, type=int,
                        help='Training epochs'
                       )       
    parser.add_argument('--pooler', default='cls', type=str,
                        help='Pooler type : {pooler_output|cls|mean|max}'
                       )    
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='Weight decay'
                       )       
    parser.add_argument('--no_decay', nargs='+', default=['bias', 'LayerNorm.weight'],
                        help='List of parameters to exclude from weight decay' 
                       )         
    parser.add_argument('--temp', default=0.05, type=float,
                        help='Temperature for similarity'
                       )       
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Drop-out ratio'
                       )       
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='Leraning rate'
                       )       
    parser.add_argument('--eta_min', default=0, type=int,
                        help='Eta min for CosineAnnealingLR scheduler'
                       )   
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon for AdamW optimizer'
                       )   
    parser.add_argument('--amp', action="store_true",
                        help='Use Automatic Mixed Precision for training'
                       ) 
    parser.add_argument('--eval_strategy', default='steps', type=str,
                        help='Evaluation strategy during training : {epoch|steps}'
                       )   
    parser.add_argument('--eval_step', default=100, type=int,
                        help='Evaluaton interval when eval_strategy is set to <steps>'
                       )   
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )
    parser.add_argument('--random_seed', default = 42, type=int,
                        help = 'Random seed'
                       )  
    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss  

def get_adamw_optimizer(model, args):
    if args.no_decay: 
        # skip weight decay for some specific parameters i.e. 'bias', 'LayerNorm'.
        no_decay = args.no_decay  
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        # weight decay for every parameter.
        optimizer_grouped_parameters = model.parameters()
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.eps)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min, last_epoch=-1)
    return scheduler

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss 

def train(encoder, train_dataloader, test_dataset, optimizer, scheduler, scaler, args):
    best_score = None
    total_train_loss = 0

    t0 = time.time()

    for epoch_i in range(args.epochs):           

        LOGGER.info(f'Epoch : {epoch_i+1}/{args.epochs}')
        
        encoder.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            # pass the data to device(cpu or gpu)            
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)

            optimizer.zero_grad()

            train_loss = encoder(input_ids, attention_mask, token_type_ids)
            
            if args.amp:
                scaler.scale(train_loss.mean()).backward()                
                scaler.step(optimizer)
                scaler.update()            

            else:
                train_loss.mean().backward()
                optimizer.step()
                
            scheduler.step()
            
            if isinstance(encoder, nn.DataParallel):
                model_to_save = encoder.module
            else:
                model_to_save = encoder
            
            if args.eval_strategy == 'steps' and (step+1) % args.eval_step == 0:
                epoch_c = round((step + 1) / len(train_dataloader), 2)
                results = inference(model_to_save, test_dataset, args)    
                spearman, pearson = results['cosine_pearson'], results['cosine_spearman']
                print(f'Epoch:{epoch_c}, Step:{step+1}, Loss:{round(float(train_loss.mean()), 4)}, Pearson:{round(pearson*100, 2)}, Spearman:{round(spearman*100, 2)}')
                    
                if not best_score or spearman > best_score:
                    best_score = spearman
                    # save_checkpoint
                    LOGGER.info(f'>>> Save the best model checkpoint in {args.output_path}.')
                    
                    if isinstance(encoder, nn.DataParallel):
                        model_to_save = encoder.module
                    else: model_to_save = encoder
                    
                    model_to_save.save_model(args.output_path)
                
                model_to_save.train()

        if args.eval_strategy == 'epoch':
            results = inference(model_to_save, test_dataset, args)    
            spearman, pearson = results['cosine_pearson'], results['cosine_spearman']
            print(f'Epoch:{epoch_i+1}, Step:{step+1}, Loss:{round(float(train_loss.mean()), 4)}, Pearson:{round(pearson*100, 2)}, Spearman:{round(spearman*100, 2)}')

            if not best_score or spearman > best_score:
                best_score = spearman
                # save_checkpoint
                LOGGER.info(f'>>> Save the best model checkpoint in {args.output_path}.')

                if isinstance(encoder, nn.DataParallel):
                        model_to_save = encoder.module
                else: model_to_save = encoder
                
                model_to_save.save_model(args.output_path)

            model_to_save.train()

    avg_train_loss = total_train_loss / (len(train_dataloader) * args.epochs)
    training_time = format_time(time.time() - t0)
    print(f"Training Time: {training_time}, Average Training Loss: {avg_train_loss}, Best Score: {round(best_score*100, 2)}")

def main(args):
    init_logging()
    seed_everything(args)
    
    LOGGER.info('*** Train SimCSE ***')
    if args.train_data.split('.')[-1] == 'txt':
        train_dataset = Dataset_TXT.load_dataset(args.train_data)
    else:
        train_dataset = Dataset_CSV.load_dataset(args.train_data)

    LOGGER.info(f'Evaluating Performance with \'{args.valid_data}\'')
    valid_dataset = Dataset_STS.load_dataset(dataset=args.valid_data)
        
    collator = DataCollator(args)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  collate_fn=collator)

    if args.device == 'cuda':
        encoder = SimCSE(args).to(args.device)
        encoder = torch.nn.DataParallel(encoder)
        LOGGER.info("Using nn.DataParallel")
    else:
        encoder = SimCSE(args).to(args.device)

    optimizer = get_adamw_optimizer(encoder, args)
    scheduler = get_scheduler(optimizer, args)

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # Train!
    torch.cuda.empty_cache()
    train(encoder, train_dataloader, valid_dataset, optimizer, scheduler, scaler, args)

if __name__ == '__main__':
    args = argument_parser()
    main(args)
