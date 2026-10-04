# decision-tree-vs-neural-network

# A Comparative Analysis of Decision Tree and Neural Network Classifiers for Iris Flower Classification

A machine learning research project comparing the performance, interpretability, and behavior of a Decision Tree classifier and a Neural Network classifier on the classic Iris flower dataset, implemented in Python using scikit-learn.

## Overview

This project trains and evaluates two fundamentally different supervised learning algorithms — a CART-based Decision Tree and a Multi-Layer Perceptron Neural Network — on the same dataset, to compare their accuracy, interpretability, and training efficiency.

Full write-up (Version 2.0, 13-page PDF) available in [Decision_Tree_vs_Neural_Network_Research_Paper.pdf](./paper/Decision_Tree_vs_Neural_Network_Research_Paper.pdf). It includes the theoretical background, methodology, model visualizations, feature importance, predicted probabilities, and limitations.

## Dataset

The [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) (Fisher, 1936): 150 samples across 3 species (setosa, versicolor, virginica), 4 numerical features (sepal length/width, petal length/width), perfectly class-balanced (50 samples per species).

## Key Results

The following results are reported in Version 2.0 of the paper, evaluated on 30 test samples:

| Metric             | Decision Tree | Neural Network |
|--------------------|---------------|----------------|
| Accuracy           |     100%      |       97%      |
| Training Time      |    0.0119s    |     0.3764s    |   
| Misclassifications |       0       |        1       |

The paper's 97% accuracy is rounded from 29 correct predictions out of 30 (approximately 96.67%). **Current code update:** `Neural_Network.ipynb` now uses a `StandardScaler` + `MLPClassifier` pipeline; its saved outputs show **100% accuracy**, **0 misclassifications**, and **0.4463 s** training time. These results differ from the paper's reported run. Training times depend on the machine and run.

**Decision Tree feature importances:** petal length (0.906) was overwhelmingly the most predictive feature, followed by petal width (0.077), sepal width (0.017), and sepal length (0.000) — showing that petal measurements dominate this fitted tree's feature importance.

![Decision Tree Visualization](./images/Decision%20Tree%20Diagram.png)
![Decision Tree Confusion Matrix](./images/Decision%20Tree%20confusion%20matrix.png)


The Neural Network's single misclassification in the paper involved a versicolor sample predicted as virginica. The confusion matrix and Softmax chart below correspond to the paper; the current scaled notebook has different saved outputs. The loss curve is a supplementary repository figure and is not included in the paper.

![Neural Network Loss Curve](./images/Neural%20Network%20loss%20curve.png)
![Neural Network Confusion Matrix](./images/Neural%20Network%20confusion%20matrix.png)
![Softmax Output — Predicted Probability Distribution](./images/Softmax%20output%20chart.png)

## Repository Structure

```text
├── notebooks/
│ ├── Scikit_learn.ipynb                         # Decision Tree implementation and evaluation
│ ├── Neural_Network.ipynb                       # Neural Network implementation and evaluation
│ └── Activation_Function_Neural_Network .ipynb      # Activation function theory and visualization
├── images/                                      # Exported figures used in the paper and README
├── paper/                                       # Full written research paper
└── README.md
```

## Tools & Libraries

- Python
- scikit-learn (`DecisionTreeClassifier`, `MLPClassifier`)
- pandas, NumPy
- Matplotlib

## Methodology

Both model notebooks use an identical 80:20 train/test split (`random_state=42`). The Decision Tree uses scikit-learn's default CART implementation with Gini Impurity. The Neural Network uses a single hidden layer of 10 neurons with ReLU activation, trained for up to 1000 iterations. The current Neural Network code standardizes features with `StandardScaler` and manually verifies the forward pass using the trained weights, ReLU, and Softmax against `predict_proba`. The paper describes the same split ratio and network architecture but does not explicitly document feature scaling or the random seed.

Evaluation covers accuracy, precision, recall, F1-score, confusion matrices, and training time. Version 2.0 also discusses unrestricted Decision Tree growth as an overfitting risk and a Neural Network convergence warning within 1000 iterations in the reported experiment. Its conclusions apply to the small Iris experiment, rather than establishing a universally superior classifier. Full methodology and discussion are detailed in the paper.

## Author

Jiwesh Mahato

## License

This project is licensed under the MIT License — see [LICENSE](./LICENSE) for details.
