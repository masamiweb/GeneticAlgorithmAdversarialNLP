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
- [GLUE](https://gluebenchmark.com)
---
## Adversarial Attack Models 
- [BERT-Attack](https://github.com/LinyangLee/BERT-Attack) [^1]
- [TextFooler](https://github.com/jind11/TextFooler) [^2]
- [BAE](https://github.com/QData/TextAttack) [^3]
- [GBDA](https://github.com/facebookresearch/text-adversarial-attack) [^4]
- [GA/alzantot](https://github.com/QData/TextAttack) [^5]
---
### Text Classification Models
- wordLSTM [^6]
- wordCNN [^7]
- BERT [^8]
---
## Colab notebooks
- 
---
## Refs
- [HuggingFace - Datasets](https://huggingface.co/datasets)
- [PapersWithCode - Datasets](https://paperswithcode.com/task/text-classification)

---

[^1]: Li, Linyang, Ruotian Ma, Qipeng Guo, Xiangyang Xue, and Xipeng Qiu. "BERT-ATTACK: Adversarial Attack against BERT Using BERT." In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6193-6202. 2020.
[^2]: Jin, Di, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolovits. "Is bert really robust? a strong baseline for natural language attack on text classification and entailment." In Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 05, pp. 8018-8025. 2020.
[^3]: Garg, Siddhant, and Goutham Ramakrishnan. "BAE: BERT-based Adversarial Examples for Text Classification." In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6174-6181. 2020.
[^4]: Guo, Chuan, Alexandre Sablayrolles, Hervé Jégou, and Douwe Kiela. "Gradient-based Adversarial Attacks against Text Transformers." arXiv preprint arXiv:2104.13733 (2021).
[^5]: Alzantot, Moustafa, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, and Kai-Wei Chang. "Generating Natural Language Adversarial Examples." In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2890-2896. 2018.
[^6]: Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9, no. 8 (1997): 1735-1780.
[^7]: Kim, Yoon. “Convolutional Neural Networks for Sentence Classification.” EMNLP (2014).
[^8]: Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018). 
