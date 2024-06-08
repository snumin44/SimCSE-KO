# 🍊 SimCSE-KO

- Lighter code to train __Korean SimCSE__   
- Dependency Problem Alleviation & Ease of Customization     
- Reference : SimCSE Paper, Original Code      

## 1. Training

- __Model__:
  - klue/bert-base
  - klue/roberta-base
- __Dataset__:
  - KorNLI (supervised)
  - Korean Wiki Text 1M (unsupervised)
- __Setting__:
  - epoch: 1
  - max length: 64
  - batch size: 256
  - learning rate: 5e-5
  - drop out: 0.1
  - temp: 0.05
  - pooler: cls
  - 1 V100 GPU 

Run Supervised Training
```python
```
Run Unsupervised Training
```python
```

## 2. Performance
- klue/bert-base
  
|Model|AVG|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Sparman|Manhatten Pearson|Manhatten Spearman|Dot Pearson|Dot Spearman|
|:---:|---|---|---|---|---|---|---|---|---|
|SimCSE-KO<br>(unsup)|||
|SimCSE-KO<br>(sup)|||

- klue/roberta-base

|Model|AVG|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Sparman|Manhatten Pearson|Manhatten Spearman|Dot Pearson|Dot Spearman|
|:---:|---|---|---|---|---|---|---|---|---|
|SimCSE-KO<br>(unsup)|||
|SimCSE-KO<br>(sup)|||

## 3. Example

## Citing
