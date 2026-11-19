# CineScope - WEKA Experimental Evaluation

## Dataset
- Dataset: cinescope_movie_ratings
- Original dataset: 100,000 instances
- Attributes: 24
- Target attribute: rating_class
- Evaluation sample: 4,999 instances

## 1. Classification Algorithms

| Metric | J48 | Random Forest | Naive Bayes | SMO |
|---|---:|---:|---:|---:|
| Accuracy (%) | 53.0106 | 52.4905 | 53.8708 | 55.2110 |
| Precision | 0.475 | 0.492 | 0.478 | 0.428 |
| Recall | 0.530 | 0.525 | 0.539 | 0.552 |
| F-measure | 0.483 | 0.501 | 0.466 | 0.413 |
| Kappa | 0.1029 | 0.1335 | 0.0971 | 0.0213 |
| MAE | 0.3709 | 0.3642 | 0.3799 | 0.3592 |
| RMSE | 0.4725 | 0.4573 | 0.4440 | 0.4594 |

### Classification observations
- SMO achieved the highest accuracy of 55.211%.
- Random Forest achieved the highest F-measure of 0.501.
- Naive Bayes achieved the lowest RMSE of 0.444.
- Random Forest achieved the highest Kappa statistic of 0.1335.
- The relatively low Kappa values indicate limited agreement beyond chance.

## 2. Clustering Algorithms

| Metric | SimpleKMeans | EM |
|---|---:|---:|
| Instances | 4999 | 4999 |
| Number of clusters | 3 | 3 |
| Correctly clustered (%) | 42.0084 | 44.6489 |
| Incorrectly clustered (%) | 57.9916 | 55.3511 |
| Build time (seconds) | 0.14 | 0.51 |

### Clustering observations
- EM achieved higher agreement with the known classes.
- SimpleKMeans required less model-building time.
- Neither clustering algorithm achieved strong class agreement.

## 3. Overall Conclusion

Four supervised classification algorithms and two unsupervised clustering algorithms were evaluated using the CineScope movie-rating dataset.

SMO achieved the highest classification accuracy, while Random Forest provided the highest F-measure. Naive Bayes achieved the lowest RMSE.

Among the clustering algorithms, EM performed better in terms of agreement with the known rating classes, whereas SimpleKMeans had a shorter model-building time.

The results demonstrate that algorithm performance varies depending on the evaluation metric. No single algorithm was best across all metrics.

## 4. Evaluation Notes

- Classification results are from the completed 4,999-instance experiments.
- SMO's completed metrics were preserved separately because a later 100,000-instance run was cancelled.
- The clustering experiments used classes-to-clusters evaluation on the training data.
- Clustering agreement percentages should not be interpreted as directly equivalent to classification accuracy.
- The reported results should be interpreted in the context of the sampled dataset and selected WEKA configurations.
