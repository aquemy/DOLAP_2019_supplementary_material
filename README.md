# Supplementary material description

This page contains additional details about the experimental setup and results discussed in the paper *Data Pipeline Selection and Optimization* submitted for the 21st International Workshop On Design, Optimization, Languages and Analytical Processing of Big Data ([DOLAP 2019](http://www.cs.put.poznan.pl/events/DOLAP2019.html)) collocated with [EDBT/ICDT](http://edbticdt2019.inesc-id.pt/?main) joint conference.


**Table of Content:**   

1. [Experiments short description](#experiments-short-description)
2. [Experiment 1: SMBO for DPSO](#experiment-1-smbo-for-dpso)
	1. [Detailed experimental protocol](#details-experimental-protocol)
	2. [Measures and analysis](#measures-and-analysis)
	3. [Results](#results)
3. [Experiment 2: Algorithm-specific Configuration](#experiment-2-algorithm-specific-configuration)
	1. [Detailed experimental protocol](#details-experimental-protocol-1)
	2. [Measures and analysis](#measures-and-analysis-1)
	3. [Results](#results-1)
	4. [NMAD Calculation](#nmad-calculation)

# Experiments short description

The paper contains two experiments:

1. Study of Sequential Model Based Optimizatoin (SMBO) to the Data Pipeline Selection and Optimization. 
2. Study of if an optimal pipeline configuration is specific to an algorithm or general to the dataset.

# Experiment 1: SMBO for DPSO
## Detailed experimental protocol

- **Datasets:** [Breast](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html), [Wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html).
- **Methods:** [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [Neural Network](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
- **Dataset split:** 60% for training set, 40% for test set.
- **Pipeline configuration space size:** 4750 configurations.

For each dataset, there is a *baseline* pipeline consisting in not doing any preprocessing.

### Step 1

For each dataset and method, we performed an exhaustive search on the configuration space defined in details right after.
For each of the **4750** configurations, a **10-fold cross-validation** has been performed and the score measure is the **accuracy**.

### Step 2

For each dataset and method, we performed a search using Sequential Model-Based Optimization (implementation provided by ```hyperopt```) and a budget of **100** configurations to visit (about 2% of the whole configuration space). As for Step 1., we measure the **accuracy** obtained over a **10-fold cross-validation**.

### Measures and analysis

We want:

- **(Q1)** to quantify the achievable improvement compared to the baseline pipeline.
- **(Q2)** to measure how likely it is to improve the baseline score according to the configuration space.
- **(Q3)** to determine if SMBO is capable to improving the baseline score.
- **(Q4)** to measure how much SMBO is likely to improve the baseline score with a restricted budget.
- **(Q5)** to measure how fast SMBO is likely to improve the baseline score with a restricted budget.

To answer those questions, we generate two kind of plots:

1. **Density of the configuration depending on the accuracy for the exhaustive grid, and for the SMBO search.** If the density is not null for accuracy higher than the baseline score, then, there exist configurations that improve the baseline score (answer to **Q1**). We can observe the probability to improve the baseline score (and quantify how much) by observing the proportion of the area after the baseline score vertical marker (answer to **Q2**). Similarely, if the density for SMBO has some support higher than the baseline score, it means SMBO search could improve (answer to **Q3**). If the area above the baseline score vertical marker is larger for SMBO than for the exhaustive search, then SMBO is more likely to improve the baseline than an exhaustive search (answer to **Q4**).
2. **Evolution of score obtained configuration after configuration for SMBO search.** The *improvement interval* is comprised between the baseline score and the best score obtained by the exhaustive search. To answer **Q5**, we plot horizontally the improvement interval, and plot the best score obtained iteration after iteration. SMBO improved the baseline as soon as the best score enters the improvement interval. To help visualization, we plot veritically the number of configurations needed to enter the interval and the number of configurations to visit before reaching the best score obtained over the budget of 100 configurations. For both market, the lower the better.

## Pipeline prototype and configuration

The configuration space for the pipeline is composed of three operations. For each operations there are 4, 5 and 4 possible operators.
Each operator has between 0 and 3 specific parameter(s). For each parameter, there is between 2 and 4 possible values. The final pipeline configuration space has a total of **4750** possible configurations.

### Pipeline prototype

The pipeline *prototype* is composed of three sequential operations:

1. **rebalance:** to handle imbalanced dataset with oversampling or downsampling techniques.
2. **normalizer:** to normalize or scale features.
3. **features:** to select the most important features or reduce the input vector space dimension.

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/pipeline.png" width="75%"  style="display:block;text-align:center"/>
<figcaption>Pipeline illustration</figcaption>
</figure>

### Pipeline operators

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

### Pipeline operator specific configuration

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

## Results

The results are sorted by dataset.

### Breast dataset 

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Breast_RandomForest.png" width="50%" />
<figcaption>Configuration density depending on accuracy - Random Forest</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Breast_RandomForest.png" width="50%" />
<figcaption>SMBO results - Random Forest</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Breast_DecisionTree.png" width="50%" />
<figcaption>Configuration density depending on accuracy - Decision Tree</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Breast_DecisionTree.png" width="50%" />
<figcaption>SMBO results - Decision Tree</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Breast_NeuralNet.png" width="50%" />
<figcaption>Configuration density depending on accuracy - Neural Net</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Breast_NeuralNet.png" width="50%" />
<figcaption>SMBO results - Neural Net</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Breast_SVM.png" width="50%" />
<figcaption>Configuration density depending on accuracy - SVM</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Breast_SVM.png" width="50%" />
<figcaption>SMBO results - SVM</figcaption>
</figure>

### Iris dataset

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Iris_RandomForest.png" width="50%" />
<figcaption>Configuration density depending on accuracy - Random Forest</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Iris_RandomForest.png" width="50%" />
<figcaption>SMBO results - Random Forest</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Iris_DecisionTree.png" width="50%" />
<figcaption>Configuration density depending on accuracy - Decision Tree</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Iris_DecisionTree.png" width="50%" />
<figcaption>SMBO results - Decision Tree</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Iris_NeuralNet.png" width="50%" />
<figcaption>Configuration density depending on accuracy - Neural Net</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Iris_NeuralNet.png" width="50%" />
<figcaption>SMBO results - Neural Net</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Iris_SVM.png" width="50%" />
<figcaption>Configuration density depending on accuracy - SVM</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Iris_SVM.png" width="50%" />
<figcaption>SMBO results - SVM</figcaption>
</figure>

### Wine dataset

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Wine_RandomForest.png" width="50%" />
<figcaption>Configuration density depending on accuracy - Random Forest</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Wine_RandomForest.png" width="50%" />
<figcaption>SMBO results - Random Forest</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Wine_DecisionTree.png" width="50%" />
<figcaption>Configuration density depending on accuracy - Decision Tree</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Wine_DecisionTree.png" width="50%" />
<figcaption>SMBO results - Decision Tree</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Wine_NeuralNet.png" width="50%" />
<figcaption>Configuration density depending on accuracy - Neural Net</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Wine_NeuralNet.png" width="50%" />
<figcaption>SMBO results - Neural Net</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/distribution_Wine_SVM.png" width="50%" />
<figcaption>Configuration density depending on accuracy - SVM</figcaption>
</figure>
<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/histogram_Wine_SVM.png" width="50%" />
<figcaption>SMBO results - SVM</figcaption>
</figure>

# Experiment 2: Algorithm-specific Configuration

## Detailed experimental protocol

## Results

### ECHR dataset

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/heatmap_ECHR_RandomForest.png" width="50%" />
<figcaption>Heatmap of the accuracy depending on the configuration - Random Forest</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/heatmap_ECHR_DecisionTree.png" width="50%" />
<figcaption>Heatmap of the accuracy depending on the configuration - Decision Tree</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/heatmap_ECHR_NeuralNet.png" width="50%" />
<figcaption>Heatmap of the accuracy depending on the configuration - Neural Net</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/heatmap_ECHR_SVM.png" width="50%" />
<figcaption>Heatmap of the accuracy depending on the configuration - SVM</figcaption>
</figure>

### Newsgroup dataset

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/heatmap_News_RandomForest.png" width="50%" />
<figcaption>Heatmap of the accuracy depending on the configuration - Random Forest</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/heatmap_News_DecisionTree.png" width="50%" />
<figcaption>Heatmap of the accuracy depending on the configuration - Decision Tree</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/heatmap_News_NeuralNet.png" width="50%" />
<figcaption>Heatmap of the accuracy depending on the configuration - Neural Net</figcaption>
</figure>

<figure>
<img src="https://aquemy.github.io/DOLAP_2019_supplementary_material/images/heatmap_News_SVM.png" width="50%" />
<figcaption>Heatmap of the accuracy depending on the configuration - SVM</figcaption>
</figure>

## NMAD calculation

### ECHR


| Method | ![](https://latex.codecogs.com/gif.latex?(n,k))  |
|---|---|
| Decision Tree | ![](https://latex.codecogs.com/gif.latex?p_1=(1,0.5)) |
| Neural Network | ![](https://latex.codecogs.com/gif.latex?p_2=(1,0.5)) |
| Random Forest | ![](https://latex.codecogs.com/gif.latex?p_3=(0.5,0.1)), ![](https://latex.codecogs.com/gif.latex?p_4=(0.75,0.1)), ![](https://latex.codecogs.com/gif.latex?p_5=(1,0.5)) |
| Linear SVM | ![](https://latex.codecogs.com/gif.latex?p_6=(0.5,0.5)), ![](https://latex.codecogs.com/gif.latex?p_7=(0.75,0.5)), ![](https://latex.codecogs.com/gif.latex?p_8=(1,0.5)) |


| Sample ECHR |
|---|
| ![](https://latex.codecogs.com/gif.latex?S(p_1)=S(p_2)=S(p_5)=S(p_8)=\\{&space;p_1,&space;p_2,&space;p_5,&space;p_8&space;\\}) |
| ![](https://latex.codecogs.com/gif.latex?S(p_3)=&space;\\{&space;p_1,&space;p_2,&space;p_3,&space;p_6&space;\\}) |
| ![](https://latex.codecogs.com/gif.latex?S(p_4)=&space;\\{&space;p_1,&space;p_2,&space;p_4,&space;p_7&space;\\}) |
| ![](https://latex.codecogs.com/gif.latex?S(p_6)=&space;\\{&space;p_1,&space;p_2,&space;p_3,&space;p_6&space;\\}) |
| ![](https://latex.codecogs.com/gif.latex?S(p_7)=&space;\\{&space;p_1,&space;p_2,&space;p_5,&space;p_7&space;\\}) |

| Point | NMAD |
|---|---|
| ![](https://latex.codecogs.com/gif.latex?(5,50000)) | 0 |
| ![](https://latex.codecogs.com/gif.latex?(3,10000)) | 0.275 |
| ![](https://latex.codecogs.com/gif.latex?(4,10000)) | 0.213 |
| ![](https://latex.codecogs.com/gif.latex?(3,50000)) | 0.175 |
| ![](https://latex.codecogs.com/gif.latex?(4,50000)) | 0.094 |


### Newsgroup


| Method | ![](https://latex.codecogs.com/gif.latex?(n,k))  |
|---|---|
| Decision Tree | ![](https://latex.codecogs.com/gif.latex?p_1=(0.75,0.05)), ![](https://latex.codecogs.com/gif.latex?p_2=(0.75,1.0)) |
| Neural Network | ![](https://latex.codecogs.com/gif.latex?p_3=(1.0,0.50)) |
| Random Forest | ![](https://latex.codecogs.com/gif.latex?p_4=(0.5,0.10)) |
| Linear SVM | ![](https://latex.codecogs.com/gif.latex?p_5=(0.25,1.0)) |


| Sample Newsgroup |
|---|
| ![](https://latex.codecogs.com/gif.latex?S(p_1)=S(p_4)=\\{p_1,p_3,p_4,p_5\\}) |
| ![](https://latex.codecogs.com/gif.latex?S(p_2)=S(p_3)=S(p_5)=\\{p_2,p_3,p_4,p_5\\}) |


| Point | NMAD |
|---|---|
| ![](https://latex.codecogs.com/gif.latex?(4,5000)) | 0.306 |
| ![](https://latex.codecogs.com/gif.latex?(4,100000)) | 0.300 |
| ![](https://latex.codecogs.com/gif.latex?(5,50000)) | 0.356 |
| ![](https://latex.codecogs.com/gif.latex?(3,10000)) | 0.294 |
| ![](https://latex.codecogs.com/gif.latex?(2,100000)) | 0.362 |



# References

[1]	I. Mani, I. Zhang. “kNN approach to unbalanced data distributions: a case study involving information extraction,” In Proceedings of workshop on learning from imbalanced datasets, 2003.    
[2]	P. Hart, “The condensed nearest neighbor rule,” In Information Theory, IEEE Transactions on, vol. 14(3), pp. 515-516, 1968.    
[3]	N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of artificial intelligence research, 321-357, 2002.    