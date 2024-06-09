import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import (
            paired_cosine_distances,
            paired_euclidean_distances,
            paired_manhattan_distances,
)

def inference(encoder, test_dataset, args):
    encoder.eval()

    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, config=config)
    
    sent0 = test_dataset.sent0
    sent1 = test_dataset.sent1
    score = test_dataset.score
    assert len(sent0) == len(sent1)
    assert len(sent0) == len(score)
    
    sents = sent0 + sent1
    
    sent_embeddings = []    
    for start_index in range(0, len(sents), args.batch_size):
        # Divide Sentences into Mini-Batch
        batch = sents[start_index : start_index + args.batch_size]
           
        features = tokenizer(batch,
                             padding=args.padding,
                             max_length=args.max_length,
                             truncation=args.truncation,)
        
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction.
        with torch.no_grad():
            pooler_output = encoder.get_embeddings(input_ids=torch.tensor(features['input_ids']).to(args.device),
                                                   attention_mask=torch.tensor(features['attention_mask']).to(args.device),
                                                   token_type_ids=torch.tensor(features['token_type_ids']).to(args.device),)
        sent_embeddings.extend(pooler_output.cpu())
    
    sent_embeddings = np.asarray([emb.numpy() for emb in sent_embeddings]) 
    sent0_embeddings, sent1_embeddings, = sent_embeddings[:len(sent0)], sent_embeddings[len(sent0):]
    assert len(sent0_embeddings) == len(sent1_embeddings)

    
    # code from : https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/EmbeddingSimilarityEvaluator.py#L9
    cosine_scores = 1 - (paired_cosine_distances(sent0_embeddings, sent1_embeddings))
    euclidean_distances = -paired_euclidean_distances(sent0_embeddings, sent1_embeddings)
    manhattan_distances = -paired_manhattan_distances(sent0_embeddings, sent1_embeddings)
    dot_products = [np.dot(sent0, sent1) for sent0, sent1 in zip(sent0_embeddings, sent1_embeddings)]

    # cosine based performance
    cosine_pearson, _ = pearsonr(score, cosine_scores)
    cosine_spearman, _ = spearmanr(score, cosine_scores)

    # Euclidean based performance
    euclidean_pearson, _ = pearsonr(score, euclidean_distances)
    euclidean_spearman, _ = spearmanr(score, euclidean_distances)
    
    # Manhatten based performance
    manhatten_pearson, _ = pearsonr(score, manhattan_distances)
    manhatten_spearman, _ = spearmanr(score, manhattan_distances)
    
    # Dot product based performance
    dot_pearson, _ = pearsonr(score, dot_products)
    dot_spearman, _ = spearmanr(score, dot_products)

    # average
    avg_pearson = (cosine_pearson + euclidean_pearson + manhatten_pearson + dot_pearson) / 4
    avg_spearman = (cosine_spearman + euclidean_spearman + manhatten_spearman + dot_spearman) / 4
    avg_total = (avg_pearson + avg_spearman) / 2 
    
    all_scores = {
        'cosine_pearson':cosine_pearson,
        'cosine_spearman':cosine_spearman,
        'euclidean_pearson':euclidean_pearson,
        'euclidean_spearman':euclidean_spearman,
        'manhatten_pearson':manhatten_pearson,
        'manhatten_spearman': manhatten_spearman,
        'dot_pearson':dot_pearson,
        'dot_spearman':dot_spearman,
        'avg_pearson':avg_pearson,
        'avg_spearman':avg_spearman,
        'avg_total':avg_total,
    }
    return all_scores