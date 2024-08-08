# 🍊 SimCSE-KO

- Lighter code to train __Korean SimCSE__   
- Dependency Problem Alleviation & Ease of Customization     
- Reference : [SimCSE Paper](https://aclanthology.org/2021.emnlp-main.552/), [Original Code](https://github.com/princeton-nlp/SimCSE)

<img src="simcse.PNG" alt="example image" width="600" height="200"/>

## 1. Training

- __Model__:
  - klue/bert-base
  - klue/roberta-base
- __Dataset__:
  - KorNLI-train (supervised training)
  - Korean Wiki Text 1M (unsupervised training)
  - KorSTS-dev (evaluation)
- __Setting__:
  - epoch: 1
  - max length: 64
  - batch size: 256
  - learning rate: 5e-5
  - drop out: 0.1
  - temp: 0.05
  - pooler: cls
  - 1 A100 GPU 

## 2. Performance
- __Inference Datset__
  - KorSTS-test
  - KlueSTS-dev

- __KorSTS-test__
  
|Model|AVG|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Spearman|Manhatten Pearson|Manhatten Spearman|Dot Pearson|Dot Spearman|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SimCSE-BERT-KO<br>(unsup)|72.85|73.00|72.77|72.96|72.92|72.93|72.86|72.80|72.53|
|SimCSE-BERT-KO<br>(sup)|85.98|86.05|86.00|85.88|86.08|85.90|86.08|85.96|85.89|
|SimCSE-RoBERTa-KO<br>(unsup)|75.79|76.39|75.57|75.71|75.52|75.65|75.42|76.41|75.63|
|SimCSE-RoBERTa-KO<br>(sup)|83.06|82.67|83.21|83.22|83.27|83.24|83.28|82.54|83.03|82.92|

- __KlueSTS-dev__

|Model|AVG|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Spearman|Manhatten Pearson|Manhatten Spearman|Dot Pearson|Dot Spearman|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SimCSE-BERT-KO<br>(unsup)|65.27|66.27|64.31|66.18|64.05|66.00|63.77|66.64|64.93|
|SimCSE-BERT-KO<br>(sup)|83.96|82.98|84.32|84.32|84.30|84.28|84.20|83.00|84.29|
|SimCSE-RoBERTa-KO<br>(unsup)|80.78|81.20|80.35|81.27|80.36|81.28|80.40|81.13|80.26|
|SimCSE-RoBERTa-KO<br>(sup)|85.31|84.14|85.64|86.09|85.68|86.04|85.65|83.94|85.30|

## 3. Implementation

- __Generate Supervised Dataset__

  You can create a supervised training dataset with KorNLI by following '**data/generate_supervised_dataset.ipynb**'.

- __Download Korean Wiki Text__
```
cd data
sh download_korean_wiki_1m.sh
```
- __Download korSTS__
```
cd data
sh download_korsts.sh
```
- __Supervised Training__
```
cd train
sh run_train_supervised.sh
```
- __Unsupervised Training__
```
cd train
sh run_train_unsupervised.sh
```
- __Evaluation__
```
cd evaluation
sh run_eval.sh
```

## 4. HuggingFace Example

- Checkpoints
  - [snumin44/simcse-ko-bert-supervised](https://huggingface.co/snumin44/simcse-ko-bert-supervised)
  - [snumin44/simcse-ko-bert-unsupervised](https://huggingface.co/snumin44/simcse-ko-bert-unsupervised)
  - [snumin44/simcse-ko-roberta-supervised](https://huggingface.co/snumin44/simcse-ko-roberta-supervised)
  - [snumin44/simcse-ko-roberta-unsupervised](https://huggingface.co/snumin44/simcse-ko-roberta-unsupervised)

- Example
```python
import numpy as np
from transformers import AutoModel, AutoTokenizer

model_path = 'snumin44/simcse-ko-bert-supervised'
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

query = '내일 아침에 비가 올까요?'

targets = [
    '내일 아침에 우산을 챙겨야 합니다.',
    '어제 저녁에는 비가 많이 내렸습니다.',
    '청계천은 대한민국 서울에 있습니다.',
    '이번 주말에 축구 대표팀 경기가 있습니다.',
    '저는 매일 아침 일찍 일어나 책을 읽습니다.'
]

query_feature = tokenizer(query, return_tensors='pt')
query_outputs = model(**query_feature, return_dict=True)
query_embeddings = query_outputs.pooler_output.detach().numpy().squeeze()

def cos_sim(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

for idx, target in enumerate(targets):
    target_feature = tokenizer(target, return_tensors='pt')
    target_outputs = model(**target_feature, return_dict=True)
    target_embeddings = target_outputs.pooler_output.detach().numpy().squeeze()
    similarity = cos_sim(query_embeddings, target_embeddings)
    print(f"Similarity between query and target {idx}: {similarity:.4f}")
```
```
Similarity between query and target 0: 0.7864
Similarity between query and target 1: 0.5695
Similarity between query and target 2: 0.2646
Similarity between query and target 3: 0.3055
Similarity between query and target 4: 0.3738
```

## Citing
```
@article{gao2021simcse,
   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
@article{ham2020kornli,
 title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
 author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
 journal={arXiv preprint arXiv:2004.03289},
 year={2020}
}
```
## Acknowledgement
This project was inspired by the work from [KoSimCSE](https://github.com/BM-K/KoSimCSE-SKT?tab=readme-ov-file).
