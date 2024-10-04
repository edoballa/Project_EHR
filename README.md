# Project_EHR
## Overview
This project utilizes a Triplet Network, implemented using PyTorch, to generate feature embeddings, which are then classified using an AdaBoost classifier. The model is applied to the FIMMG dataset to predict the onset of Type 2 Diabetes (T2D), evaluating three distinct cases based on different subsets of features. The goal is to assess the model's ability to predict diabetes using varying levels of information from Electronic Health Records (EHR).

## Dataset
**To obtain the FIMMG dataset in Excel format, request the VRAI research team at the following link:** https://vrai.dii.univpm.it/content/fimmg-dataset

The project works with the FIMMG dataset, which contains medical and demographic data. We evaluate the model under three different cases, each focusing on a specific subset of features.

### Cases:
1. **Case I:**
   - **Description:** This case uses all available features from the EHR, including demographic, clinical, and monitoring information.
   - **Objective:** To evaluate the model's performance using the full range of available data.
2. **Case II:**
   - **Description:** In this case, a subset of features collected before the clinical diagnosis of Type 2 Diabetes (T2D) is used. This excludes features that could indicate a known diagnosis, such as prescriptions and exemptions specific to T2D.
   - **Objective:** To predict the onset of T2D using only data available before the confirmed diagnosis.
3. **Case III:**
   - **Description:** This case follows the same approach as Case II but limits the sample to subjects between the ages of 60 and 80. This aims to control for chronological age, ensuring that age-related factors don't influence the model.
   - **Objective:** To evaluate the model's performance in predicting T2D while controlling for age as a variable.
  
## How it works
1. **Triplet Network Training:** The Triplet Network learns to generate meaningful embeddings by minimizing the distance between similar samples (positive pairs) and maximizing the distance between dissimilar samples (negative pairs) using triplet loss.
2. **Embedding Generation:** After training, the model generates embeddings for the dataset, which represent the feature space in a more compact and discriminative form.
3. **AdaBoost Classification:** The generated embeddings are used to train an AdaBoost classifier to predict whether a subject has Type 2 Diabetes. The model is evaluated using cross-validation on several metrics.

## Dependencies
- Python 3.7+
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- TQDM
