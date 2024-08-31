import matplotlib.pyplot as plt

def plot_elbow(cluster_errors):
    plt.plot(range(2, 11), cluster_errors, "o-")
    plt.xlabel("No. Clusters")
    plt.ylabel("SSE")
    plt.title("Elbow Method")
    plt.show()

def plot_silhouette(silhouette_scores):
    plt.plot(range(2, 11), silhouette_scores, "o-")
    plt.xlabel("No. Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores for Different Numbers of Clusters")
    plt.show()

# src/evaluate.py
from pycaret.classification import plot_model

def analyze_model_performance(tuned_model):
    plot_model(tuned_model)
    plot_model(tuned_model, plot='learning')
    plot_model(tuned_model, plot='error')
    plot_model(tuned_model, plot='confusion_matrix')
    plot_model(tuned_model, plot='feature')


