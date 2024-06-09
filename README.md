# 🍊 SimCSE-KO

- Lighter code to train __Korean SimCSE__   
- Dependency Problem Alleviation & Ease of Customization     
- Reference : SimCSE Paper, Original Code

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
  - 1 V100 GPU 

## 2. Performance
- __Inference Datset__
  - KorSTS-test
  - KlueSTS-dev

- __KorSTS-test__
  
|Model|AVG|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Spearman|Manhatten Pearson|Manhatten Spearman|Dot Pearson|Dot Spearman|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SimCSE-BERT-KO<br>(unsup)|66.68|67.24|66.62|66.31|66.65|66.24|66.52|67.24|66.64|
|SimCSE-BERT-KO<br>(sup)|82.22|81.63|82.52|82.39|82.57|82.33|82.52|81.50|82.34|
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

  You can create a supervised training dataset with KorNLI by following 'data/generate_supervised_dataset.ipynb'.

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

## Citing
