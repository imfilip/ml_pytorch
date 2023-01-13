"""
Basic utils functions
"""
from typing import Any
import numpy.typing as npt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def addd(a: int, b: int) -> int:
    """
    Add two integers.

    Parameters
    
    """
    return a + b


def plot_decision_regions(X: npt.ArrayLike, y: npt.ArrayLike, classifier: Any, resolution: float = 0.02):
    """Plot decision boundaries.
    Args:
        X: Training set.
        y: Target values.
        classifier: Model used to perform classification task with predict() method implemented.
        resolution: Resolution used to build grid points. Defaults to 0.02.
    Returns:
        None. Shows plot.
    """

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    markers =  ('o', 's', '^', 'v', '<')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
    x2_min, x2_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha = 0.2, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, label in enumerate(np.unique(y)):
        plt.scatter(x=X[y == label, 0],
        y=X[y == label, 1],
        alpha=0.8,
        c=colors[idx],
        marker=markers[idx],
        label=f'Class {label}',
        edgecolor='black')

    return None
