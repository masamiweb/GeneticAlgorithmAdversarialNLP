# Vulnerability of Natural Language Classifiers to Evolutionary Generated Adversarial Text

## ToDo (last updated: 2021W44)
- [Mohamed] Paper section 2 (related work) + investigating implementaiton
- [Sandy] Paper section 5 (results) + investigating implementaiton
- [Mani] Check sections 3-5 for other resources (refs, datasets, etc...) 
---
## Datasets
- [IMDB link1](https://datasets.imdbws.com) -- [IMDB link2](http://ai.stanford.edu/~amaas/data/sentiment/)
- [Yelp link1](https://www.yelp.com/dataset) -- [Yelp link2](https://www.kaggle.com/yelp-dataset/yelp-dataset)
- [AG News](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
- [Movie Reviews - Rotten Tomatoes](https://www.cs.cornell.edu/people/pabo/movie-review-data/)
- [SST-2 - GLUE](https://gluebenchmark.com)
- [FAKE](https://www.kaggle.com/c/fake-news/data) >>> no pre-trained models
---
### [Text Classification Models](https://textattack.readthedocs.io/en/latest/3recipes/models.html)
- wordLSTM [^1]
- wordCNN [^2]
- BERT [^3]
---
## Adversarial Attack Models 
- [BERT-Attack](https://github.com/LinyangLee/BERT-Attack) [^4]
- [TextFooler](https://github.com/jind11/TextFooler) [^5]
- [BAE](https://github.com/QData/TextAttack) [^6]
- [GBDA](https://github.com/facebookresearch/text-adversarial-attack) [^7] --> NOT POSSIBLE, complicated to use!
- [GA/alzantot](https://github.com/QData/TextAttack) [^8]
- [CLARE](https://github.com/QData/TextAttack) [^9]
---
## Evaluation Metrics
- Table: black-box attack evaluation (datasets, text classification models, attack models, original accuracy, adversarial accuracy, % Perturbed Word, Semantic Similarity, Query Number, Average Text Length)
- Table: runtime comparison (attack models vs datasets)
- Table: datasets stats
- Graph: attack effectiveness over dataset (x-axis: max % perturbation, y-axis: test accuracy, lines: attack models)
- Table: Qualitative examples (attack models vs datasets) showing replacements and insertions
---
## Colab Notebooks
- [Playground](https://colab.research.google.com/drive/1Hs_E6F0_h5AYhUj3o5wNI5LeF5Ashk8p?usp=sharing)
- [TextAttack-Playground-CMD](https://colab.research.google.com/drive/1rRdiD5oQy_ohHrIDF4Nsal7Fdom-Q2D-?usp=sharing)
- [TextAttack-Playground-API](https://colab.research.google.com/drive/1uU4xYNGfpvv-H2eirRr9U67GYTkMoBnm?usp=sharing)
---
## Refs
- [HuggingFace - Datasets](https://huggingface.co/datasets)
- [PapersWithCode - Datasets](https://paperswithcode.com/task/text-classification)

---

[^1]: Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9, no. 8 (1997): 1735-1780.
[^2]: Kim, Yoon. “Convolutional Neural Networks for Sentence Classification.” EMNLP (2014).
[^3]: Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018). 
[^4]: Li, Linyang, Ruotian Ma, Qipeng Guo, Xiangyang Xue, and Xipeng Qiu. "BERT-ATTACK: Adversarial Attack against BERT Using BERT." In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6193-6202. 2020.
[^5]: Jin, Di, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolovits. "Is bert really robust? a strong baseline for natural language attack on text classification and entailment." In Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 05, pp. 8018-8025. 2020.
[^6]: Garg, Siddhant, and Goutham Ramakrishnan. "BAE: BERT-based Adversarial Examples for Text Classification." In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6174-6181. 2020.
[^7]: Guo, Chuan, Alexandre Sablayrolles, Hervé Jégou, and Douwe Kiela. "Gradient-based Adversarial Attacks against Text Transformers." arXiv preprint arXiv:2104.13733 (2021).
[^8]: Alzantot, Moustafa, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, and Kai-Wei Chang. "Generating Natural Language Adversarial Examples." In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2890-2896. 2018.
[^9]: Li, Dianqi, Yizhe Zhang, Hao Peng, Liqun Chen, Chris Brockett, Ming-Ting Sun, and William B. Dolan. "Contextualized Perturbation for Textual Adversarial Attack." In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 5053-5069. 2021.
