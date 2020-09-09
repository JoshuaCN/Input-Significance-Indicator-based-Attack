# Adversarial Attack by Input Significance Indicator
A fast and efficient white-box iterative adversarial attack algorithm against deep learning models based on score backpropagation.

## Introduction
This repository contains the code for the real data experiments of the paper:
- [Generating Adversarial Examples with Input Significance Indicator](https://doi.org/10.1016/j.neucom.2020.01.040)

## Method
By backdistributing confidence scores of some image through the model, we get every input feature a sensitivity or relevance score, which depends on rules of backpropagation.
We then iteratively find and perturb the most significant feature until the termination condition is reached.

## Dependencies
- tensorflow
- [Innvesitigate](https://github.com/albermax/innvestigate)(Fast implementation of significance score backpropagation)
- [adversarial robustness toolbox](https://github.com/IBM/adversarial-robustness-toolbox) (for evaluation)
