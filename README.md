# QualifAI
### Cambridge Computer Science Tripos - Part II Project
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](markdown/PartIIDissertation-11.05.23-mct52.pdf)

### TL:DR
QualifAI takes a data input and, in real time, trains a Gaussian Process in order to predict qualifying cut-off times useful to a Formula One team. QualifAI results in higher accuracy and higher confidence results for what time is likely to be accepted for a given qualifying session

<details>
<summary> <b> Abstract</b> (click me to read)</summary>
<p>
This project is inspired by the growing need for fast, accurate decision-making within every aspect of the Formula One weekend. It is assumed that every team will have software that allows them to predict many aspects of how the race weekend will unfold, including qualifying cut-off times, and as such it is important that every team is consistently upgrading these predictors in order to increase the likelihood that they can generate a correct prediction. While a new qualifying cut-off time predictor was made recently for the client this did not give the probability of a given driver passing into the next session, and only the cut-off time for that session. This project builds on the previous iteration by focusing on the probabilistic aspect of models to allow for informed decisions. 
</p>
</details>

## Description

This project aims to complete two goals, predicting cutoff times for a given qualifying sessions and predicting the likelihood that a driver will continue into the next session given that they do not complete any more laps of the circuit. This project is completed in collaboration with Mercedes-AMG Petronas Formula One Team.

This is done via Machine Learning techniques, namely a Gaussian Process Regressor, trained on previous races in order to predict statistics about current sessions.

A pipeline is created such that data can be easily extracted from the client supplied API and used in a model.

## Results
Results are extensively analysed in chapter 4 of the paper. A summary of results is given below.
![mae results.png](markdown%2Fmae%20results.png)

![error range results.png](markdown%2Ferror%20range%20results.png)

![comparison with deliver.png](markdown%2Fcomparison%20with%20deliver.png)