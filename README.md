# Adversarial-Learning-with-FGSM-attacks-and-OOD-Detection
This repository contains the implementation and exploration of adversarial learning with Fast Gradient Sign Method augmented training set and Out Of Distribution Detection tasks.

# Description of the project 

This repository showcases the implementation and exploration of adversarial attacks and out-of-distribution (OOD) detection techniques in PyTorch.
The work encompasses three main experiments to understand adversarial vulnerabilities, robust model training, and targeted attacks.

All the results are in `README.md` file.

1. Experiment 1: Out-of-Distribution (OOD) Detection
I began by implementing OOD detection using the "max logits" method. By thresholding the maximum softmax output of a neural network, I did not effectively detect OOD samples in a given dataset. I then extended the analysis to include additional evaluation metrics such as Area Under the Curve (AUC) and Precision-Recall (PR) curves to assess the performance more comprehensively.

2. Experiment 2: Fast Gradient Sign Method (FGSM) Attack and Robust Model Training
In this experiment, I delved into adversarial attacks by implementing the Fast Gradient Sign Method (FGSM). I trained a robust model by augmenting the training set with perturbed samples generated through FGSM. This process exposed the vulnerability of neural networks to adversarial perturbations and allowed me to enhance model robustness.

3. Experiment 3: Targeted FGSM Attack and Model Evaluation
In the final experiment, I extended FGSM to a targeted version, enabling to create adversarial samples imitating specific target classes. These targeted samples were then used to evaluate both a standard and a robust model. The results provided insights into how well each model could resist targeted attacks.

# Results Analysis

# Experiment 1.1: Build a simple OOD detection pipeline

In this experiment I build a simple Out-Of-Distribution (OOD) detection pipeline. 
I am using as **ID** dataset CIFAR-10 and as **OOD** dataset a subset of CIFAR-100 containing the following classes: ['bottle', 'clock', 'plate', 'telephone'].

For calculating the OOD detection, I used *max logist* as in the suggestion, to verify if it was a good discriminator or not. 
The following graph shows the results: 

![OOD logits](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_1_ood_logits.png)

As anticipated by the lesson, there is a lot of overlapping between the 2 curves, even if we did not expect that. 
Infact, I selected categories different from the CIFAR-10 dataset to create a subset of CIFAR-100 with the goal to have a more different results. 
This results confirm what has been told during the lesson, hence max logits is not a good discriminative metric for this kind of task.

# Experiment 1.2: Measure your OOD detection performance

I am using two threshold-free approaches: the area under the Receiver Operator Characteristic (ROC) curve for ID classification, and the area under the Precision-Recall curve for both ID and OOD scoring.

## 1.2.1: ROC and AUC metric

![auc](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_1_auc.png)

The achieved AUC of 0.63 indicates that the Out-of-Distribution (OOD) detection system is capable of distinguishing between in-distribution and out-of-distribution samples with moderate effectiveness. 
An AUC value closer to 1 would indicate a stronger discriminative ability. 
Although the current AUC suggests some level of differentiation, there is room for improvement to enhance the system's performance.

## 1.2.1: Precision-Recall metric 

![pr](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_1_pr.png)

The computed AP of 0.89 implies that the OOD detection system is effective in ranking the relevance of detected out-of-distribution samples during the precision-recall analysis.
An AP value closer to 1 signifies better precision and recall balance. 
The high AP value indicates the system's ability to achieve both high precision (low false positive rate) and high recall (low false negative rate).

# Experiment 2.1: Enhancing Robustness to Adversarial Attack

In thi exercise I am implementing Fast Gradient Sign Method (FGSM) method to generate some attacks to the samples. 
Recal that the FGSM perturbs samples in the direction of the gradient with respect to the input x. 

In this exercise I am performing a qualitative and a quantitative evaluation of the performance of the previous model (standard model). 

Before starting, recall that the **test accuracy = 60%** of the standard model.

## Experiment 2.1.1: Qualitative Evaluation 

In this section I will perform a qualitative evaluation of the FGSM attacked samples, with 3 different values of epsilon: 0.01, 0.1, 03. 

We will see in the quantitative evaluation, that just a value of 0.02 is already enough to ruin the image at the point which the accuracy drop from 60% to about 20%.

**Before Attack** 

![before attack](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_before_attack.png)

**Epsilon = 0.01**

![fgsm_1](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_after_attack_1.png)

**Epsilon = 0.1**

![fgsm_2](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_after_attack_2.png)

**Epsilon = 0.3**

![fgsm_3](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_after_attack_3.png)

These results suggest that the attack is effective! And as the quantitative resuls will show, the quality drops exponentially. Indeed, from 0.1 to 0.3 the image is almost unrecognizible.

## Experiment 2.1.2: Quantitative Evaluation

Now I will provie a quantitative evaluation. Recall that the model accuracy before the attack was about 60%, we can see that with just an epsilon of 0.01 the accuracy dramatically drops to 30%, and as we espect, the more we grow and the worse it becomes.

![accuracy after attack](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_qe.png)

The trend drops exponentially, and eventually stabilize. The quantitative results confirm the qualitative results we have seen previously.

# Experiment 2.2: Augment training with adversarial examples

Use your implementation of FGSM to augment your training dataset with adversarial samples. Ideally, you should implement this data augmentation on the fly so that the adversarial samples are always generated using the current model. Evaluate whether the model is more (or less) robust to ID samples using your OOD detection pipeline and metrics you implemented in Exercise 1.

The trained model with samples augmented with FGSM will be called **robust model**

## Experiment 2.2.1: Quantitative Evaluation 

I wanted to perform a quantiative evaluation comparing the **robust model** with the **standard model** over the attacked samples, varying the values of epsilon. 

![comparison ](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_fgsm_qe.png)

The blue line is the standard model accuracy calculated on the samples that received a fgsm attack and it has the same trend as before, of course. 
The orange one is the other trained on fgsm attacked samples, and as we can notice, it is much more robust than the standard one. 
This shows the importance of training a model on attacked samples. 
Moreover, the model was trained with an epsilon = 0.1, but still it is robust with respect to bigger values of epsilon, while the blue one drop exponentially with greater values of epsilon.

## Experiment 2.2.2: OOD Detection Pipeline

Let's use the previous ood-pipeline to evaluate the **robust_model**

### Experiment 2.2.2.1: OOD detection with logits 

![ood](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_fgsm_ood.png)

Even if the model was trained on attacked sample, unfortunately, it does not show a very distintive difference with OOD samples. That, again, means that the max-logits is not a good way for OOD-detection.

### Experiment 2.2.2.2: ROC and AUC metrics

Let's try to use more advanced techniques, like in the previous experiment, with Area Under the Curve of ROC and the Precision-Recall curve

![auc](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_fgsm_auc.png)

The achieved AUC of 0.65 indicates a slight improvement respect the previous model. It indicates that the Out-of-Distribution (OOD) detection system is capable of distinguishing between in-distribution and out-of-distribution samples with moderate effectiveness. An AUC value closer to 1 would indicate a stronger discriminative ability. Although the current AUC suggests some level of differentiation, there is room for improvement to enhance the system's performance.

### Experiment 2.2.2.3: Precision Recall curve metrics

![pr](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_fgsm_pr.png)

The computed AP of 0.90 show a slight improvement with respect to the previous model, not so impressive. We have to admit that it was already high with the standard model. Overral, this result implies that the OOD detection system is effective in ranking the relevance of detected out-of-distribution samples during the precision-recall analysis. An AP value closer to 1 signifies better precision and recall balance. The high AP value indicates the system's ability to achieve both high precision (low false positive rate) and high recall (low false negative rate).

# Experiment 3.3: Experiment with *targeted* adversarial attacks

In this section I have implement the Fast Gradient Sign Method for generating targeted attacks, in order for the samples to imitate samples from a specific class.

I then evaluated the adversarial samples quantiatively, comparing the standard model with the robust model previously trained on the general fgsm. For doing it I have plotted the accuracy over the targeted attacked samples in function of the value of epsilon.

Then, I evaluated the adversarial samples qualitatively with 3 plots:

1. Plot of clean images with predictions of standard model.
2. Plot of perturbed images with predictions of standard model.
3. Plot of perturbed images with predictions of robust model.

All the experiments have been conducted using as target class **horse** of CIFAR-10, but you can repeat it changing the target with what you prefer.
Epsilon = 0.1.

## Experiment 3.3.1: Quantiative Evaluation 

![qe](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex3_standard_vs_robust_accuracy.png)

The graph shows once again that the **robust model** is actually more robust with respect to the perturbations. 
Infact, it didn't lose much accuracy, and the trend seems to be *linear* with the value of epsilon. 
The **standard model** instead, showed a strange path, growin in accuracy (still under the 60% on normal set), from 30% and then startin descending again.

## Experiment 3.3.2: Qualitative Evaluation 

In this section I wanted to evaluate adversarial samples qualitatively with 3 plots:

1. Plot of clean images with predictions of standard model.
2. Plot of perturbed images with predictions of standard model.
3. Plot of perturbed images with predictions of robust model.

**Plot of clean images with predictions of standard model**

![plot](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_3_plot_img_1.png)

We can see that the predictions are pretty accurate (remember that his model has a 60% test accuracy), as we expected.

**Plot of perturbed images with predictions of standard model**

![plot](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_3_plot_img_2.png)

In this case the model starts to be very wrong and misclassifies the majority of images, as expected. 

**Plot of perturbed images with predictions of robust model**

![plot](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_3_plot_img_3.png)

Even the robust_model is innacurate, but still much more accurate wrt the standard model on the perturbed images. This thanks to the adversarial training. 
