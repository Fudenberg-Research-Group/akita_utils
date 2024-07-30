import seaborn as sns
import matplotlib.pyplot as plt


# SINGLE-MAP PLOTTING FUNCTION


def plot_map(matrix, vmin=-0.6, vmax=0.6, width=5, height=5, palette="RdBu_r"):
    """
    Plot a log(obs/exp) map with reasonable defaults for AkitaV2.

    Parameters
    ------------
    matrix : numpy array
        Predicted log(obs/exp) map.
    vmin : float
    vmax : float
        Minimum and maximum in the colormap scale.
    width : int
    height : int
        Width and height of a plotted map.
    """

    fig = plt.figure(figsize=(width, height))

    sns.heatmap(
        matrix,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        cmap=palette,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.show()
