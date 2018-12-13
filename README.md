# Supplementary material description

This page contain additional details about the experimental setup and results discussed in the paper *Data Pipeline Hyperparameter Optimization* submitted for the 21st International Workshop On Design, Optimization, Languages and Analytical Processing of Big Data ([DOLAP 2019](http://www.cs.put.poznan.pl/events/DOLAP2019.html)) collocated with [EDBT/ICDT](http://edbticdt2019.inesc-id.pt/?main) joint conference.

# Experiments short description

# Detailed experimental protocol

- **Datasets:** Adult, Breast, Iris, Wine.
- **Methods:** SVM, Random Forest, Neural Network, Decision Tree.
- **Dataset split:** 60% for training set, 40% for test set.
- **Pipeline configuration space size:** 4750 configurations.

For each dataset, there is a *baseline* pipeline consisting in not doing any preprocessing.

## Step 1

For each dataset and method, we performed an exhaustive search on the configuration space defined in details right after.
For each of the **4750** configurations, a **10-fold cross-validation** has been performed and the score measure is the **accuracy**.

## Step 2

For each dataset and method, we performed a search using Sequential Model-Based Optimization (implementation provided by ```hyperopt```) and a budget of **100** configurations to visit (about 2% of the whole configuration space). As for Step 1., we measure the **accuracy** obtained over a **10-fold cross-validation**.

## Measures and analysis

We want:

- **(Q1)** to quantify the achievable improvement compared to the baseline pipeline.
- **(Q2)** to measure how likely it is to improve the baseline score according to the configuration space.
- **(Q3)** to determine if SMBO is capable to improving the baseline score.
- **(Q4)** to measure how much SMBO is likely to improve the baseline score with a restricted budget.
- **(Q5)** to measure how fast SMBO is likely to improve the baseline score with a restricted budget.

To answer those questions, we generate two kind of plots:

1. **Density of the configuration depending on the accuracy for the exhaustive grid, and for the SMBO search.** If the density is not null for accuracy higher than the baseline score, then, there exist configurations that improve the baseline score (answer to **Q1**). We can observe the probability to improve the baseline score (and quantify how much) by observing the proportion of the area after the baseline score vertical marker (answer to **Q2**). Similarely, if the density for SMBO has some support higher than the baseline score, it means SMBO search could improve (answer to **Q3**). If the area above the baseline score vertical marker is larger for SMBO than for the exhaustive search, then SMBO is more likely to improve the baseline than an exhaustive search (answer to **Q4**).
2. **Evolution of score obtained configuration after configuration for SMBO search.** The *improvement interval* is comprised between the baseline score and the best score obtained by the exhaustive search. To answer **Q5**, we plot horizontally the improvement interval, and plot the best score obtained iteration after iteration. SMBO improved the baseline as soon as the best score enters the improvement interval. To help visualization, we plot veritically the number of configurations needed to enter the interval and the number of configurations to visit before reaching the best score obtained over the budget of 100 configurations. For both market, the lower the better.

# Pipeline prototype and configuration

The configuration space for the pipeline is composed of three operations. For each operations there are 4, 5 and 4 possible operators.
Each operator has between 0 and 3 specific parameter(s). For each parameter, there is between 2 and 4 possible values. The final pipeline configuration space has a total of **4750** possible configurations.

## Pipeline prototype

The pipeline *prototype* is composed of three sequential operations:

1. **rebalance:** to handle imbalanced dataset with oversampling or downsampling techniques.
2. **normalizer:** to normalize or scale features.
3. **features:** to select the most important features or reduce the input vector space dimension.

![Pipeline illustration](/pipeline.png)

## Pipeline operators

For the step **rebalance**, the possible methods to instanciate are:

- ```None```: no sample modification.
- ```NearMiss```: Undersampling using Near Miss method [1].    
Implementation: [imblearn.under_sampling.NearMiss](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NearMiss.html#imblearn-under-sampling-nearmiss)
- ```CondensedNearestNeighbour```: Undersampling using Near Miss method [2].    
Implementation: [imblearn.under_sampling.CondensedNearestNeighbour](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.CondensedNearestNeighbour.html#imblearn-under-sampling-condensednearestneighbour)
- ```SMOTE```: Oversampling using Synthetic Minority Over-sampling Technique [3].     
Implementation: [imblearn.over_sampling.SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html#imblearn-over-sampling-smote)

For the step **normalizer**, the possible methods to instanciate are:
- ```None```: no normalization.
- ```StandardScaler```: Standardize features by removing the mean and scaling to unit variance.     
Implementation: [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn-preprocessing-standardscaler)
- ```PowerTransform```: Apply a Yeo-Johnson transformation to make data more Gaussian-like.     
Implementation: [sklearn.preprocessing.PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn-preprocessing-powertransformer)
- ```MinMaxScaler```: Transforms features by scaling each feature to [0,1].    
Implementation: [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn-preprocessing-minmaxscaler)
- ```RobustScaler```: Same as StandardScaler but remove points outside a range of percentile.    
Implementation: [sklearn.preprocessing.RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn-preprocessing-robustscaler)

For the step **features**, the possible methods to instanciate are:
- ```None```: no feature transformation.
- ```PCA```: Keep the k main axis of a Principal Component Analysis.    
Implementation: [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn-decomposition-pca)
- ```SelectKBest```: Selecting the k most informative features according to ANOVA and F-score.     
Implementation: [sklearn.feature_selection.SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn-feature-selection-selectkbest)
- ```PCA+SelectKBest```: Union of features obtaines by PCA and SelectKBest.     
Implementation: [sklearn.pipeline.FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn-pipeline-featureunion)

**REMARK:** The baseline pipeline corresponds to the triple ```(None, None, None)```.

## Pipeline operator specific configuration

- ```NearMiss```: 
	- ```n_neighbors```: [1,2,3]
- ```CondensedNearestNeighbour```:
	- ```n_neighbors```: [1,2,3]
- ```SMOTE```: 
	- ```k_neighbors```: [5,6,7]
- ```StandardScaler```: 
	- ```with_mean```: [True, False]
	- ```with_std```: [True, False]
- ```RobustScaler```: 
	- ```quantile_range```:[(25.0, 75.0),(10.0, 90.0), (5.0, 95.0)]
	- ```with_centering```: [True, False]
    - ```with_scaling```: [True, False]
- ```PCA```: 
	- ```n_components```: [1,2,3,4]
- ```SelectKBest```: 
	- ```k```: [1,2,3,4]
- ```PCA+SelectKBest```: 
	- ```n_components```: [1,2,3,4] 
	- ```k```:[1,2,3,4] 

# Results

The results are sorted by dataset.

## Adult dataset

## Breast dataset 

![Configuration density depending on accuracy - Random Forest](/images/distribution_Breast_RandomForest.png)
![SMBO results - Random Forest](/images/histogram_Breast_RandomForest.png)

![Configuration density depending on accuracy - Decision Tree](/images/distribution_Breast_DecisionTree.png)
![SMBO results - Decision Tree](/images/histogram_Breast_DecisionTree.png)

![Configuration density depending on accuracy - Neural Net](/images/distribution_Breast_NeuralNet.png)
![SMBO results - Neural Net](/images/histogram_Breast_NeuralNet.png)

## Iris dataset

![Configuration density depending on accuracy - Random Forest](/images/distribution_Iris_RandomForest.png)
![SMBO results - Random Forest](/images/histogram_Iris_RandomForest.png)

![Configuration density depending on accuracy - Decision Tree](/images/distribution_Iris_DecisionTree.png)
![SMBO results - Decision Tree](/images/histogram_Iris_DecisionTree.png)

![Configuration density depending on accuracy - Neural Net](/images/distribution_Iris_NeuralNet.png)
![SMBO results - Neural Net](/images/histogram_Iris_NeuralNet.png)

![Configuration density depending on accuracy - SVM](/images/distribution_Iris_SVM.png)
![SMBO results - SVM](/images/histogram_Iris_SVM.png)


## Wine dataset

![Configuration density depending on accuracy - Random Forest](/images/distribution_Wine_RandomForest.png)
![SMBO results - Random Forest](/images/histogram_Wine_RandomForest.png)

![Configuration density depending on accuracy - Decision Tree](/images/distribution_Wine_DecisionTree.png)
![SMBO results - Decision Tree](/images/histogram_Wine_DecisionTree.png)

![Configuration density depending on accuracy - Neural Net](/images/distribution_Wine_NeuralNet.png)
![SMBO results - Neural Net](/images/histogram_Wine_NeuralNet.png)

![Configuration density depending on accuracy - SVM](/images/distribution_Wine_SVM.png)
![SMBO results - SVM](/images/histogram_Wine_SVM.png)

# References

[1]	I. Mani, I. Zhang. “kNN approach to unbalanced data distributions: a case study involving information extraction,” In Proceedings of workshop on learning from imbalanced datasets, 2003.    
[2]	P. Hart, “The condensed nearest neighbor rule,” In Information Theory, IEEE Transactions on, vol. 14(3), pp. 515-516, 1968.    
[3]	N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of artificial intelligence research, 321-357, 2002.    