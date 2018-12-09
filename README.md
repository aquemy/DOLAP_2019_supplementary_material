# Supplementary material description

This page contain additional details about the experimental setup and results discussed in the paper *Data Pipeline Hyperparameter Optimization* submitted for the 21st International Workshop On Design, Optimization, Languages and Analytical Processing of Big Data ([DOLAP 2019](http://www.cs.put.poznan.pl/events/DOLAP2019.html)) collocated with [EDBT/ICDT](http://edbticdt2019.inesc-id.pt/?main) joint conference.

# Experiments short description

# Detailed experimental protocol

# Pipeline prototype and configuration

The configuration space for the pipeline is composed of three operations. For each operations there are 4, 5 and 4 possible operators.
Each operator has between 0 and 3 specific parameter(s). For each parameter, there is between 2 and 4 possible values. The final pipeline configuration space has a total of **4750** possible configurations.

## Pipeline prototype

The pipeline *prototype* is composed of three sequential operations:

1. **rebalance:** to handle imbalanced dataset with oversampling or downsampling techniques.
2. **normalizer:** to normalize or scale features.
3. **features:** to select the most important features or reduce the input vector space dimension.

## Pipeline operators

For the step **rebalance**, the possible methods to instanciate are:

- ```None```: no sample modification.
- ```NearMiss```: Undersampling using Near Miss method [1]. Implementation: [imblearn.under_sampling.NearMiss](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NearMiss.html#imblearn-under-sampling-nearmiss)
- ```CondensedNearestNeighbour```: Undersampling using Near Miss method [1]. Implementation: [imblearn.under_sampling.NearMiss](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NearMiss.html#imblearn-under-sampling-nearmiss)
- ```SMOTE```: Oversampling using Synthetic Minority Over-sampling Technique [3]. Implementation: [imblearn.over_sampling.SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html#imblearn-over-sampling-smote)

For the step **normalizer**, the possible methods to instanciate are:
- ```None```: no normalization.
- ```StandardScaler```: 
- ```PowerTransform```: 
- ```MinMaxScaler```: 
- ```RobustScaler```: 

For the step **features**, the possible methods to instanciate are:
- ```None```: no feature transformation.
- ```PCA```: Principal Component Analysis
- ```SelectKBest```: Selecting the k most informative features according to ANOVA and Fisher score
- ```PCA+SelectKBest```: Union of features obtaines by PCA and SelectKBest.

## Pipeline operator specific configuration



# Results

The results are sorted by dataset.

## Adult dataset

## Breast dataset 

![Configuration density depending on accuracy - Random Forest](/images/distribution_Breast_RandomForest.png)
![SMBO results - Random Forest](/images/histogram_Breast_RandomForest.png)

## Iris dataset

![Configuration density depending on accuracy- Random Forest](/images/distribution_Iris_RandomForest.png)
![SMBO results - Random Forest](/images/histogram_Iris_RandomForest.png)

## Wine dataset

![Configuration density depending on accuracy - Random Forest](/images/distribution_Wine_RandomForest.png)
![SMBO results - Random Forest](/images/histogram_Wine_RandomForest.png)


# References

[1]	I. Mani, I. Zhang. “kNN approach to unbalanced data distributions: a case study involving information extraction,” In Proceedings of workshop on learning from imbalanced datasets, 2003.    
[2]	P. Hart, “The condensed nearest neighbor rule,” In Information Theory, IEEE Transactions on, vol. 14(3), pp. 515-516, 1968.    
[3]	N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of artificial intelligence research, 321-357, 2002.    