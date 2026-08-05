# decision-tree-vs-neural-network

# A Comparative Analysis of Decision Tree and Neural Network Classifiers for Iris Flower Classification

A machine learning research project comparing the performance, interpretability, and behavior of a Decision Tree classifier and a Neural Network classifier on the classic Iris flower dataset, implemented in Python using scikit-learn.

## Overview

This project trains and evaluates two fundamentally different supervised learning algorithms — a CART-based Decision Tree and a Multi-Layer Perceptron Neural Network — on the same dataset, under identical experimental conditions, to compare their accuracy, interpretability, and training efficiency.

Full write-up available in [Decision_Tree_vs_Neural_Network_Research_Paper.pdf](./paper/Decision_Tree_vs_Neural_Network_Research_Paper.pdf).

## Dataset

The [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) (Fisher, 1936): 150 samples across 3 species (setosa, versicolor, virginica), 4 numerical features (sepal length/width, petal length/width), perfectly class-balanced (50 samples per species).

## Key Results

| Metric             | Decision Tree | Neural Network |
|--------------------|---------------|----------------|
| Accuracy           |     100%      |       97%      |
| Training Time      |    0.0119s    |     0.3764s    |   
| Misclassifications |       0       |        1       |

**Decision Tree feature importances:
**petal length (0.906) was overwhelmingly the most predictive feature, followed by petal width (0.077), sepal width (0.017), and sepal length (0.000) — confirming that petal measurements alone are nearly sufficient to classify Iris species.

![Decision Tree Visualization](./images/Decision%20Tree%20Diagram.png)
![Decision Tree Confusion Matrix](./images/Decision%20Tree%20confusion%20matrix.png)


The Neural Network's single misclassification involved a versicolor sample predicted as virginica — the two most visually similar species in the dataset.

![Neural Network Loss Curve](./images/Neural%20Network%20loss%20curve.png)
![Neural Network Confusion Matrix](./images/Neural%20Network%20confusion%20matrix.png)
![Softmax Output — Predicted Probability Distribution](./images/Softmax%20output%20chart.png)

## Repository Structure
├── notebooks/
│ ├── Scikit_learn.ipynb                         # Decision Tree implementation and evaluation
│ ├── Neural_Network.ipynb                       # Neural Network implementation and evaluation
│ └── Activation_Function_Neural_Network.ipynb      # Activation function theory and visualization
├── images/                                      # Exported figures used in the paper and README
├── paper/                                       # Full written research paper
└── README.md

## Tools & Libraries

- Python
- scikit-learn (`DecisionTreeClassifier`, `MLPClassifier`)
- pandas, NumPy
- Matplotlib

## Methodology

Both models were trained on an identical 80:20 train/test split (`random_state=42`) to ensure a fair comparison. The Decision Tree used scikit-learn's default CART implementation with Gini Impurity. The Neural Network used a single hidden layer of 10 neurons with ReLU activation, trained for up to 1000 iterations. Full methodology detailed in the paper.

## Author

Jiwesh Mahato

## License

This project is licensed under the MIT License — see [LICENSE](./LICENSE) for details.
