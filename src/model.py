import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast 
from transformers import AutoConfig, AutoModel, AutoTokenizer

class MLPLayer(nn.Module):

    def __init__(self, dropout, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features):
        x = self.dense(features)
        x = self.dropout(x)
        x = self.activation(x)
        return x

class Similarity(nn.Module):

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    
    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        # 'cls before pooler' of original code corresponds to 'cls'.
        assert self.pooler_type in ['pooler_output', 'cls', 'mean', 'max']
        
    def forward(self, attention_mask, outputs):
        last_hidden_state = outputs.last_hidden_state
        # hidden_states = outputs.hidden_states

        if self.pooler_type == 'pooler_output':
            return outputs.pooler_output
        
        elif self.pooler_type == 'cls':
            return last_hidden_state[:,0,:]
        
        # code from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py#L9-L241 
        elif self.pooler_type == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask  # mean_embeddings : (batch_size, hidden_size)
            return mean_embeddings 

        # code from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py#L9-L241
        elif self.pooler_type == 'max':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_embeddings = torch.max(last_hidden_state, 1)[0] # max_embeddings : (batch_size, hidden_size)
            return max_embeddings

        else:
            raise NotImplementedError

class SimCSE(nn.Module):

    def __init__(self, args):
        super(SimCSE, self).__init__()
        self.config = AutoConfig.from_pretrained(args.model)
        self.encoder = AutoModel.from_pretrained(args.model, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, config=self.config)
        self.encoder.resize_token_embeddings(len(self.tokenizer))        
        self.device = args.device

        self.mlp = MLPLayer(args.dropout, self.config)
        self.sim = Similarity(args.temp)
        self.pooler = Pooler(args.pooler)
    
    @autocast()
    def forward(self, input_ids, attention_mask, token_type_ids):
        # input_ids, attention_mask, token_type_ids : (batch_size, num_sent, hidden_size)
        batch_size, num_sent = input_ids.size(0), input_ids.size(1)

        # Flatten input features for encoding.
        # input_ids, attention_mask, token_type_ids : (batch_size * num_sent, hidden_size)
        input_ids = input_ids.view((-1, input_ids.size(-1)))
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) 
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))

        outputs =  self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        pooler_output = self.pooler(attention_mask, outputs)
        # pooler_output : (batch_size, num_sent, hidden_size)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))
                        
        # all outputs except the 'pooler_output' are passed through the mlp layer.
        if self.pooler in ['cls', 'mean', 'max']:
            pooler_output = self.mlp(pooler_output)
        
        # seperate representations
        # z1, z2, z3 : (batch_size, hidden_size)
        z1, z2 = pooler_output[:,0], pooler_output[:,1]
        if num_sent == 3:
            z3 = pooler_output[:,2]

        # calculate cosine similarity
        total_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        if num_sent == 3:
            z1_z3_sim = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            total_sim = torch.cat([total_sim, z1_z3_sim], 1)

        # set labels and loss function
        labels = torch.arange(total_sim.size(0)).long().to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(total_sim, labels)
        return loss

    def get_embeddings(self, input_ids, attention_mask, token_type_ids):
        outputs =  self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_output = self.pooler(attention_mask, outputs)
        return pooler_output

    def save_model(self, path):
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)