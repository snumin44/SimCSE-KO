# 🍊 SimCSE-KO

- Lighter code to train __Korean SimCSE__   
- Dependency Problem Alleviation & Ease of Customization     
- Reference : SimCSE Paper, Original Code      

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
  - 1 V100 GPU 

- Run Supervised Training
```python
```
- Run Unsupervised Training
```python
```

## 2. Performance
- __Inference Datset__
  - KorSTS-test
  - KlueSTS-dev

- __KorSTS-test__
  
|Model|AVG|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Sparman|Manhatten Pearson|Manhatten Spearman|Dot Pearson|Dot Spearman|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SimCSE-BERT-KO<br>(unsup)|||
|SimCSE-BERT-KO<br>(sup)|82.22|81.63|82.52|82.39|82.57|82.33|82.52|81.50|82.34|
|SimCSE-RoBERTa-KO<br>(unsup)|||
|SimCSE-RoBERTa-KO<br>(sup)|83.06|82.67|83.21|83.22|83.27|83.24|83.28|82.54|83.03|82.92|

- __KlueSTS-dev__

|Model|AVG|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Sparman|Manhatten Pearson|Manhatten Spearman|Dot Pearson|Dot Spearman|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SimCSE-BERT-KO<br>(unsup)|||
|SimCSE-BERT-KO<br>(sup)|83.96|82.98|84.32|84.32|84.30|84.28|84.20|83.00|84.29|
|SimCSE-RoBERTa-KO<br>(unsup)|||
|SimCSE-RoBERTa-KO<br>(sup)|85.31|84.14|85.64|86.09|85.68|86.04|85.65|83.94|85.30|

## 3. Example

## Citing
