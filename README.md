# Evolutionary Ensemble Learning (EEL) for
## Binary Neural Networks (BNN)


**Why?**

* With evolutionary training there is no need for derivable functions. Therefore there is no need for continuous values. The BNNs always keep their values binary (weights and activations), leading to lighter computations.
* Evolutionary training naturally works with ensemble methods because there is no need to train each weak learner separately. We can keep the last offspring as the classifier ensemble (Post Mortem Ensemble Creation).
* BNN are faster than classical NN but very unstable. Grouping them with bagging or boosting improve the stability of the classifier while keeping it light (in term of memory and computation).
* Can be parallelized.


### Some references

* [Generic Methods for Evolutionary Ensemble Learning](https://pdfs.semanticscholar.org/4160/3b18538cab6f198d896f5ea6a8f37a091d88.pdf) - Christian Gagné, Michèle Sebag, Marc Schoenauer, Marco Tomassini.
* [Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit?](https://arxiv.org/pdf/1806.07550.pdf) - Shilin Zhu, Xin Dong, Hao Su.

