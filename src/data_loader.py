import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

class Dataset_CSV(Dataset):

    def __init__(self, sent0, sent1, hard_neg=None):
        self.sent0 = sent0
        self.sent1 = sent1
        self.hard_neg = hard_neg

    @classmethod
    def load_dataset(cls, data_path, delimiter='\t'):
        df = pd.read_csv(data_path, sep=delimiter)
        
        if len(df.columns) == 1:
            sent0 = df['sent0'].to_list()
            sent1 = df['sent0'].to_list()  
            return cls(sent0, sent1)
        
        elif len(df.columns) == 2:
            sent0 = df['sent0'].to_list()
            sent1 = df['sent1'].to_list()
            return cls(sent0, sent1)

        elif len(df.columns) == 3:
            sent0 = df['sent0'].to_list()
            sent1 = df['sent1'].to_list()
            hard_neg = df['hard_neg'].to_list()
            return cls(sent0, sent1, hard_neg)

        else:
            raise NotImplementedError

    def __len__(self):
        assert len(self.sent0) == len(self.sent1)
        if self.hard_neg: 
            assert len(self.sent0) == len(self.hard_neg)
        return len(self.sent0)

    def __getitem__(self, index):
        if self.hard_neg: 
            return {'sent0':self.sent0[index],
                    'sent1':self.sent1[index],
                    'hard_neg':self.hard_neg[index]}
        
        else:
            return {'sent0':self.sent0[index],
                    'sent1':self.sent1[index]}


class Dataset_TXT(Dataset):

    def __init__(self, sent0, sent1, hard_neg=None):
        self.sent0 = sent0
        self.sent1 = sent1
        self.hard_neg = hard_neg

    @classmethod
    def load_dataset(cls, data_path, delimiter='\t'):
        
        with open(data_path, 'r') as file:
            sents = file.readlines()
        columns = len(sents[0].split(delimiter))
        
        if columns == 1:
            sent0, sent1 = sents, sents
            return cls(sent0, sent1)
        
        elif columns == 2:
            sent0, sent1 = [], []
            for sent in sents:
                sent_pair = sent.split(delimiter)
                sent0.append(sent_pair[0])
                sent1.append(sent_pair[1])
            return cls(sent0, sent1)

        elif columns == 3:
            sent0, sent1, hard_neg = [], [], []
            for sent in sents:
                sent_triplet = sent.split(delimiter)
                sent0.append(sent_triplet[0])
                sent1.append(sent_triplet[1])  
                har_neg.append(sent_triplet[2])
            return cls(sent0, sent1, hard_neg)

        else:
            raise NotImplementedError

    def __len__(self):
        assert len(self.sent0) == len(self.sent1)
        if self.hard_neg: 
            assert len(self.sent0) == len(self.hard_neg)
        return len(self.sent0)

    def __getitem__(self, index):
        if self.hard_neg: 
            return {'sent0':self.sent0[index],
                    'sent1':self.sent1[index],
                    'hard_neg':self.hard_neg[index]}
        
        else:
            return {'sent0':self.sent0[index],
                    'sent1':self.sent1[index]}


class Dataset_STS(object):

    def __init__(self, sent0, sent1, score):
        self.sent0 = sent0
        self.sent1 = sent1
        self.score = score

    @classmethod
    def load_dataset(cls, dataset='klue_dev'):
        # KLUE STS Dataset : train / validation
        if 'klue' in dataset:
            if 'train' in dataset:
                split = 'train'
            else: split = 'validation'
        
            klue_sts = load_dataset('klue', 'sts', trust_remote_code=True)
            train_set = klue_sts[split]
            sent0_lst, sent1_lst, score_lst = [], [], []
            for sample in train_set:
                sent0_lst.append(sample['sentence1'])
                sent1_lst.append(sample['sentence2'])
                score_lst.append(sample['labels']['label'])
            return cls(sent0_lst, sent1_lst, score_lst)
        
        # KorSTS Datset : train / dev / test
        elif 'kor' in dataset:
            if 'train' in dataset:
                file_name = '../data/sts-train.tsv'
            elif 'test' in dataset:
                file_name = '../data/sts-test.tsv'
            else: file_name = '../data/sts-dev.tsv'
        
            # origianl sts-dev.tsv file cannot be opened with pandas.read_csv
            with open(file_name, 'r') as file:
                samples = file.readlines()[1:] # remove header

            sent0_lst, sent1_lst, score_lst = [], [], []
            for sample in samples:
                columns = sample.split('\t')
                sent0 = columns[-2].replace(',', '').replace('\'', '').replace('\"', '')
                sent1 = columns[-1].replace(',', '').replace('\'', '').replace('\"', '').replace('\n', '')
    
                sent0_lst.append(sent0)
                sent1_lst.append(sent1)
                score_lst.append(float(columns[-3]))
            return cls(sent0_lst, sent1_lst, score_lst)           
        
        else:
            raise NotImplementedError


class DataCollator(object):
    
    def __init__(self, args):
        self.config = AutoConfig.from_pretrained(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                       config=self.config)
        self.padding = args.padding
        self.max_length = args.max_length
        self.truncation = args.truncation

    def __call__(self, samples):
        sent0_lst = []
        sent1_lst = []
        hard_negs = []
        for sample in samples:
            sent0_lst.append(sample['sent0'])        
            sent1_lst.append(sample['sent1'])
            if 'hard_neg' in sample.keys():
                hard_negs.append(sample['hard_neg'])

        sents_num = len(samples)
        sents_lst = sent0_lst + sent1_lst + hard_negs

        # Encode all sentences at once as a single encoder. 
        sent_features = self.tokenizer(
            sents_lst,
            padding = self.padding,
            max_length = self.max_length,
            truncation = self.truncation
            )
        
        features = {}
        if len(hard_negs) == 0: 
            for key in sent_features:
                features[key] = [[sent_features[key][i],
                                  sent_features[key][i+sents_num]] for i in range(sents_num)]            

        else:   
            for key in sent_features:
                features[key] = [[sent_features[key][i],
                                  sent_features[key][i+sents_num],
                                  sent_features[key][i+sents_num *2]] for i in range(sents_num)] 
                
        batch = {
            'input_ids':torch.tensor(features['input_ids']),
            'attention_mask':torch.tensor(features['attention_mask']),
            'token_type_ids':torch.tensor(features['token_type_ids']),        
            } 
        return batch