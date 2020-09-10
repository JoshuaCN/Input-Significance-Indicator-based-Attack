## Adversarial Attack based on Input Significance Indicator
A fast and efficient white-box iterative adversarial attack algorithm against deep learning models.
This repository contains the implementation code for the paper:
- [Generating Adversarial Examples with Input Significance Indicator](https://doi.org/10.1016/j.neucom.2020.01.040)

## Method
By backpropagating confidence scores of some image through the model with certain rule, we get every input feature a score signifying its importance from some perspective. 


We can then iteratively find and perturb the most significant feature until the termination condition is reached, which leads to an adversarial attack aiming for the least changed input elements, in other words, an ![1](http://latex.codecogs.com/svg.latex?l_0) constrained adversarial attack. 


With simple modifications, this attack can be extended to other norms, such as ![2](http://latex.codecogs.com/svg.latex?l_2) (by perturbing multiple features at each iteration) and ![3](http://latex.codecogs.com/svg.latex?l_\infty) (by perturbing all features with a small value according to their significance).


Currently two indicators are supported, including **input sensitivity** and **input relevance**.

**Sensitivity** measures how much changes in each feature will affect the final classification, can be derived by taking the derivative of logits with respect to each input element. 

**Relevance** quantifies how much each feature contributes to the final classification, this is done by back-decompositing the final logits layer by layer, until each input element is assigned a relevance score, called [layer-wise relevance propagation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).

## Main Files
```
create_model.py
```
used for model definition.
```
isi_attack.py
```
main implementation of isi attack.
```
eval.py
```
contains functions for evaluation or visualization.
```
notebook
```
example notebook that shows how the attack works, including a comparison with JSMA, can be runned in google colab.
## Dependencies
- tensorflow=1.13.1
- [iNNvesitigate](https://github.com/albermax/innvestigate)(Fast implementation of significance score backpropagation)
