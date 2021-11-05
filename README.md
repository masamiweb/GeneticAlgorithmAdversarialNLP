# Vulnerability of Natural Language Classifiers to Evolutionary Generated Adversarial Text

## ToDo (last updated: 2021W44)
- [Mohamed] Paper section 2 (related work) + investigating implementaiton
- [Sandy] Paper section 5 (results) + investigating implementaiton
- [Mani] Check sections 3-5 for other resources (refs, datasets, etc...) 
---
## Datasets
- [IMDB link1](https://datasets.imdbws.com) -- [IMDB link2](http://ai.stanford.edu/~amaas/data/sentiment/)
- [Yelp link1](https://www.yelp.com/dataset) -- [Yelp link2](https://www.kaggle.com/yelp-dataset/yelp-dataset)
- [FAKE](https://www.kaggle.com/c/fake-news/data)
- [AG News](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
---
### Implementations
- [BERT-Attack](https://github.com/LinyangLee/BERT-Attack) [^1]
---
### Colab notebooks
- 
---
## Refs
- [HuggingFace - Datasets](https://huggingface.co/datasets)
- [PapersWithCode - Datasets](https://paperswithcode.com/task/text-classification)

---


Computing Science honours dissertation project aimed at generating adversarial examples against a sentiment analysis model, using genetic algorithms. 

### To run the code, run the Jupyter notebooks in the following order
(You will need a machine with a decent NVIDIA GPU to build and test the model)
- 01-Build_model_and_Distance_Matrix.ipynb
- 02-Generate_Attacks.ipynb
- 03-Stats.ipynb


[^1]: Li, Linyang, Ruotian Ma, Qipeng Guo, Xiangyang Xue, and Xipeng Qiu. "BERT-ATTACK: Adversarial Attack against BERT Using BERT." In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6193-6202. 2020.
