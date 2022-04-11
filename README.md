# Vulnerability of Natural Language Classifiers to Evolutionary Generated Adversarial Text

## [Results in Google spreadsheet](https://docs.google.com/spreadsheets/d/1n1tqPyxtPwdFt8XGoS9Cc_D5nrercAXx1P8PfqgLwzk/edit?usp=sharing)
## https://github.com/thunlp/OpenAttack#usage-examples

## ToDo (last updated: 2022W4)
- Experiments
  - Datasets = ["mr", "ag-news", "imdb", "yelp"]
  - Classification_Models = ["lstm","bert", "cnn"]
  - Attacks = ["bae", "a2t", "textfooler", "textbugger", "##ours##"]
- [Mohamed] Finish experiments and start paper writing (e.g. tables, metrics).
- [Sandy] Finish proposed implementation.
- [Mani] Do proof reading.
---
## Datasets
- [IMDB link1](https://datasets.imdbws.com) -- [IMDB link2](http://ai.stanford.edu/~amaas/data/sentiment/)
- [Yelp link1](https://www.yelp.com/dataset) -- [Yelp link2](https://www.kaggle.com/yelp-dataset/yelp-dataset)
- [AG News](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
- [Movie Reviews - Rotten Tomatoes](https://www.cs.cornell.edu/people/pabo/movie-review-data/)
- [SST-2 - GLUE](https://gluebenchmark.com)
- [FAKE](https://www.kaggle.com/c/fake-news/data)
---
### [Text Classification Models](https://textattack.readthedocs.io/en/latest/3recipes/models.html)
- wordLSTM [^1]
- wordCNN [^2]
- BERT [^3]
---
## Adversarial Attack Models 
- [BERT-Attack - 2019](https://github.com/LinyangLee/BERT-Attack) [^4]
- [TextFooler - 2019](https://github.com/jind11/TextFooler) [^5]
- [BAE - 2019](https://github.com/QData/TextAttack) [^6]
- [GBDA - 2021](https://github.com/facebookresearch/text-adversarial-attack) [^7] 
- [GA/alzantot - 2018](https://github.com/QData/TextAttack) [^8]
- [CLARE - 2020](https://github.com/QData/TextAttack) [^9]
- [TextBugger - 2018](https://github.com/QData/TextAttack) [^10]
- [A2T - 2021](https://github.com/QData/TextAttack) [^11]
- [fast-alzantot - 2019](https://github.com/QData/TextAttack) [^12]
---
## Evaluation Metrics
- Table: black-box attack evaluation (datasets, text classification models, attack models, original accuracy, adversarial accuracy, % Perturbed Word, Semantic Similarity, Query Number, Average Text Length)
- Table: runtime comparison (attack models vs datasets)
- Table: datasets stats
- Graph: attack effectiveness over dataset (x-axis: max % perturbation, y-axis: test accuracy, lines: attack models) --> Not possible!! Needs edits over attack implementations.
- Table: Qualitative examples (attack models vs datasets) showing replacements and insertions
---
## Colab Notebooks
- [Playground](https://colab.research.google.com/drive/1Hs_E6F0_h5AYhUj3o5wNI5LeF5Ashk8p?usp=sharing)
- [TextAttack-Playground-CMD](https://colab.research.google.com/drive/1rRdiD5oQy_ohHrIDF4Nsal7Fdom-Q2D-?usp=sharing)
- [TextAttack-Playground-API](https://colab.research.google.com/drive/1uU4xYNGfpvv-H2eirRr9U67GYTkMoBnm?usp=sharing)
- [TextAttack-Playground-API-V2](https://colab.research.google.com/drive/1seoSdC419jxFsJotr3m39AIhjDuV21co?usp=sharing)
- [NLP_TextAttack_Test_api--V3--SB-MN](https://colab.research.google.com/drive/17bstCgQ8TPocFXRVUtHW4NsOD2kyYJO9?usp=sharing) --> Issue in replace=False
- [NLP_TextAttack_Test_api--V4--SB-MN](https://colab.research.google.com/drive/1meFoEkyU_e4MCUIamoK25l942ONHtAXF#scrollTo=obkXVaBm0sVG?usp=sharing) --> ManjinderMethodB contraints changed, changed max_pop to 20 and iteration to 30, and distance measure to use max_mse_dist=0.5 - use mse distance because it cheaper than using cosine distance


## Notes:
- [results](https://drive.google.com/drive/folders/1jnI7Tqe-zkJEIScX-vYyj-c2bGGpx5FZ?usp=sharing)
- [Attack] GA/alzantot -- 2018 : Very slow to execute! (removed)
- [Attack] GBDA -- 2021 : NOT POSSIBLE, complicated to use! (removed)
- [Attack] CLARE - 2020 : GPU out of memory error! (removed)
- [Attack] fast-alzantot -- 2019 : still very slow! (removed)
- [Dataset] yelp : very big dataset, taking forever to run! (removed)
- [Dataset] FAKE : no pre-trained models! (removed)
- [[bug]NLP_TextAttack_Test_api--Bug-mr__cnn__a2t](https://colab.research.google.com/drive/10pV0ArRPIG0DjmgPjIXZhldCLRAciYbP?usp=sharing) -- [reported bug](https://github.com/QData/TextAttack/issues/601)
- [[semi-bug]NLP_TextAttack_Test_api--Bug-mr__lstm__bert-attack](https://colab.research.google.com/drive/1Efk4dp9gHrtvTkSIv_KxY8l6J808BCx1?usp=sharing) -- under investigation!!
- TO RUN:
  -  ag-news__ bert __ ** : done!
  -  imdb __ bert __ ** : done!
  -  yelp __ ** __ ** : done!
  -  ** __ ** __ bert-attack : error (freeze)!
  -  ** __ cnn __ a2t : error to fix!

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
[^10]: Li, J., S. Ji, T. Du, B. Li, and T. Wang. "TextBugger: Generating Adversarial Text Against Real-world Applications." In 26th Annual Network and Distributed System Security Symposium. 2019.
[^11]: Yoo, Jin Yong, and Yanjun Qi. "Towards Improving Adversarial Training of NLP Models." In Findings of the Association for Computational Linguistics: EMNLP 2021, pp. 945-956. 2021.
[^12]: Jia, Robin, Aditi Raghunathan, Kerem Göksel, and Percy Liang. "Certified Robustness to Adversarial Word Substitutions." In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4129-4142. 2019.
