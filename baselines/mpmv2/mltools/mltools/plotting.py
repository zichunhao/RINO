"""A collection of plotting scripts for standard uses."""

import contextlib
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.axes._axes as axes
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import binned_statistic, pearsonr

# Some defaults for my plots to make them look nicer
plt.rcParams["xaxis.labellocation"] = "right"
plt.rcParams["yaxis.labellocation"] = "top"
plt.rcParams["legend.edgecolor"] = "1"
plt.rcParams["legend.loc"] = "upper left"
plt.rcParams["legend.framealpha"] = 0.0
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["axes.titlesize"] = "large"
plt.rcParams["legend.fontsize"] = 11


def render_image(fig: plt.Figure) -> PIL.Image:
    """Render a matplotlib figure as a PIL image."""
    return PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Return only a portion of a matplotlib colormap."""
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )


def gaussian(x_data, mu=0, sig=1):
    """Return the value of the gaussian distribution."""
    return (
        1 / np.sqrt(2 * np.pi * sig**2) * np.exp(-((x_data - mu) ** 2) / (2 * sig**2))
    )


def plot_profiles(
    x_list: np.ndarray,
    y_list: np.ndarray,
    data_labels: list,
    ylabel: str,
    xlabel: str,
    central_statistic: str | Callable = "mean",
    up_statistic: str | Callable = "std",
    down_statistic: str | Callable = "std",
    bins: int | list | np.ndarray = 50,
    figsize: tuple = (5, 4),
    hist_kwargs: list | None = None,
    err_kwargs: list | None = None,
    legend_kwargs: dict | None = None,
    path: Path | None = None,
    return_fig: bool = False,
    return_img: bool = False,
) -> None:
    """Plot and save a profile plot."""
    assert len(x_list) == len(y_list)

    # Make sure the kwargs are lists too
    if not isinstance(hist_kwargs, list):
        hist_kwargs = len(x_list) * [hist_kwargs]
    if not isinstance(err_kwargs, list):
        err_kwargs = len(x_list) * [err_kwargs]

    # Initialise the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, (x, y) in enumerate(zip(x_list, y_list, strict=False)):
        # Get the basic histogram to setup the counts and edges
        hist, bin_edges = np.histogram(x, bins)

        # Get the central values for the profiles
        central = binned_statistic(x, y, central_statistic, bin_edges)
        central_vals = central.statistic

        # Get the up and down values for the statistic
        up_vals = binned_statistic(x, y, up_statistic, bin_edges).statistic
        if not (up_statistic == "std" and down_statistic == "std"):
            down_vals = binned_statistic(x, y, down_statistic, bin_edges).statistic
        else:
            down_vals = up_vals

        # Correct based on the uncertainty of the mean
        if up_statistic == "std":
            up_vals = central_vals + up_vals / np.sqrt(hist + 1e-8)
        if down_statistic == "std":
            down_vals = central_vals - down_vals / np.sqrt(hist + 1e-8)

        # Get the additional keyword arguments for the histograms
        if hist_kwargs[i] is not None and bool(hist_kwargs[i]):
            h_kwargs = deepcopy(hist_kwargs[i])
        else:
            h_kwargs = {}

        # Use the stairs function to plot the histograms
        line = ax.stairs(central_vals, bin_edges, label=data_labels[i], **h_kwargs)

        # Get the additional keyword arguments for the histograms
        if err_kwargs[i] is not None and bool(err_kwargs[i]):
            e_kwargs = deepcopy(err_kwargs[i])
        else:
            e_kwargs = {"color": line.get_edgecolor(), "alpha": 0.2, "fill": True}

        # Include the uncertainty in the plots as a shaded region
        ax.stairs(up_vals, bin_edges, baseline=down_vals, **e_kwargs)

    # Limits
    ylim1, ylim2 = ax.get_ylim()
    ax.set_ylim(top=ylim2 + 0.5 * (ylim2 - ylim1))
    ax.set_xlim([bin_edges[0], bin_edges[-1]])

    # Axis labels and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(**(legend_kwargs or {}))
    ax.grid(visible=True)

    # Final figure layout
    fig.tight_layout()

    # Save the file
    if path is not None:
        fig.savefig(path)

    # Return a rendered image, or the matplotlib figure, or close
    if return_img:
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close(fig)
        return img
    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_corr_heatmaps(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    bins: list,
    xlabel: str,
    ylabel: str,
    path: Path | None = None,
    weights: np.ndarray | None = None,
    do_log: bool = True,
    equal_aspect: bool = True,
    cmap: str = "coolwarm",
    incl_line: bool = True,
    incl_cbar: bool = True,
    title: str = "",
    figsize=(6, 5),
    do_pearson=False,
    return_fig: bool = False,
    return_img: bool = False,
) -> None:
    """Plot a heatmap of the correlation between two datasets.

    Parameters
    ----------
    x_vals : np.ndarray
        The x-values of the data points.
    y_vals : np.ndarray
        The y-values of the data points.
    bins : list
        The number of bins to use for the histogram.
        If a single value is provided, it will be used for both x and y axes.
        If a list of two values is provided, the first value will be used for
        the x-axis and the second value for the y-axis.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    path : Path | None, optional
        The path to save the plot image. If None, the plot will not be saved.
    weights : np.ndarray | None, optional
        An array of weights for each data point. If None, all data points
        will have equal weight.
    do_log : bool, optional
        Whether to use logarithmic scale for the color mapping.
    equal_aspect : bool, optional
        Whether to maintain equal aspect ratio for the plot.
    cmap : str, optional
        The colormap to use for the heatmap.
    incl_line : bool, optional
        Whether to include a line showing the range of the data.
    incl_cbar : bool, optional
        Whether to include a colorbar. (default: True)
    title : str, optional
        The title of the plot. If empty, no title will be displayed.
    figsize : tuple, optional
        The size of the figure in inches. (default: (6, 5))
    do_pearson : bool, optional
        Whether to calculate and display the Pearson correlation coefficient.
    return_fig : bool, optional
        Whether to return the figure object.
    return_img : bool, optional
        Whether to return the plot image as a PIL Image object.

    Returns
    -------
    None
        If `return_fig` and `return_img` are both False.
    matplotlib.figure.Figure
        The figure object, if `return_fig` is True.
    PIL.Image.Image
        The plot image as a PIL Image object, if `return_img` is True.
    """
    # Define the bins for the data
    if isinstance(bins, partial):
        bins = bins()
    if len(bins) != 2:
        bins = [bins, bins]

    # Initialise the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    hist = ax.hist2d(
        x_vals.flatten(),
        y_vals.flatten(),
        bins=bins,
        weights=weights,
        cmap=cmap,
        norm="log" if do_log else None,
    )
    if equal_aspect:
        ax.set_aspect("equal")

    # Add line
    if incl_line:
        ax.plot([min(hist[1]), max(hist[1])], [min(hist[2]), max(hist[2])], "k--", lw=1)

    # Add colourbar
    if incl_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # Hacky solution to fix this sometimes failing if the values are shit
        with contextlib.suppress(ValueError):
            fig.colorbar(hist[3], cax=cax, orientation="vertical", label="frequency")

    # Axis labels and titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not title:
        ax.set_title(title)

    # Correlation coeficient
    if do_pearson:
        ax.text(
            0.05,
            0.92,
            f"r = {pearsonr(x_vals, y_vals)[0]:.3f}",
            transform=ax.transAxes,
            fontsize="large",
            bbox={"facecolor": "white", "edgecolor": "black"},
        )

    # Save the image
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    if return_fig:
        return fig
    if return_img:
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close(fig)
        return img
    plt.close(fig)
    return None


def add_hist(
    ax: axes.Axes,
    data: np.ndarray,
    bins: np.ndarray,
    do_norm: bool = False,
    label: str = "",
    scale_factor: float | None = None,
    hist_kwargs: dict | None = None,
    err_kwargs: dict | None = None,
    do_err: bool = True,
) -> None:
    """Plot a histogram on a given axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot the histogram on.
    data : numpy.ndarray
        The data to plot as a histogram.
    bins : int
        The bin edges to use for the histogram
    do_norm : bool, optional
        Whether to normalize the histogram, by default False.
    label : str, optional
        The label to use for the histogram, by default "".
    scale_factor : float, optional
        A scaling factor to apply to the histogram, by default None.
    hist_kwargs : dict, optional
        Additional keyword arguments to pass to the histogram function, by default None.
    err_kwargs : dict, optional
        Additional keyword arguments to pass to the errorbar function, by default None.
    do_err : bool, optional
        Whether to include errorbars, by default True.
    """
    # Compute the histogram
    hist, _ = np.histogram(data, bins)
    hist_err = np.sqrt(hist)

    # Normalise the errors
    if do_norm:
        divisor = np.array(np.diff(bins), float) / hist.sum()
        hist = hist * divisor
        hist_err = hist_err * divisor

    # Apply the scale factors
    if scale_factor is not None:
        hist *= scale_factor
        hist_err *= scale_factor

    # Get the additional keyword arguments for the histograms
    h_kwargs = hist_kwargs if hist_kwargs is not None and bool(hist_kwargs) else {}

    # Use the stairs function to plot the histograms
    line = ax.stairs(hist, bins, label=label, **h_kwargs)

    # Get the additional keyword arguments for the error bars
    if err_kwargs is not None and bool(err_kwargs):
        e_kwargs = err_kwargs
    else:
        e_kwargs = {"color": line.get_edgecolor(), "alpha": 0.5, "fill": True}

    # Include the uncertainty in the plots as a shaded region
    if do_err:
        ax.stairs(hist + hist_err, bins, baseline=hist - hist_err, **e_kwargs)


def quantile_bins(
    data: np.ndarray,
    bins: int = 50,
    low: float = 0.001,
    high: float = 0.999,
    axis: int | None = None,
) -> np.ndarray:
    """Return bin edges between quantile values of a dataset."""
    return np.linspace(*np.quantile(data, [low, high], axis=axis), bins)


def plot_multi_correlations(
    data_list: list | np.ndarray,
    data_labels: list,
    col_labels: list,
    n_bins: int = 50,
    bins: list | None = None,
    fig_scale: float = 1,
    n_kde_points: int = 50,
    levels: int = 3,
    do_err: bool = True,
    do_norm: bool = True,
    hist_kwargs: list | None = None,
    err_kwargs: list | None = None,
    legend_kwargs: dict | None = None,
    path: Path | str | None = None,
    return_img: bool = False,
    return_fig: bool = False,
) -> plt.Figure | None:
    """Plot multiple correlations in a matrix format.

    Parameters
    ----------
    data_list : list | np.ndarray
        List of data arrays to be plotted.
    data_labels : list
        List of labels for the data.
    col_labels : list
        List of column labels for the data.
    n_bins : int, optional
        Number of bins for the histogram, by default 50. Superseeded by bins
    bins : list | None, optional
        List of bin edges, by default None.
    fig_scale : float, optional
        Scaling factor for the figure size, by default 1.
    n_kde_points : int, optional
        Number of points for the KDE plot, by default 50.
    levels : int, optional
        Number of levels for the KDE plot, by default 3.
    do_err : bool, optional
        If True, add error bars to the histogram, by default True.
    do_norm : bool, optional
        If True, normalize the histogram, by default True.
    hist_kwargs : list | None, optional
        List of dictionaries with keyword arguments for the plotting function,
        by default None.
    err_kwargs : list | None, optional
        List of dictionaries with keyword arguments for the error function,
        by default None.
    legend_kwargs : dict | None, optional
        Dictionary with keyword arguments for the legend function, by default None.
    path : Path | str, optional
        Path where to save the figure, by default None.
    return_img : bool, optional
        If True, return the image as a PIL.Image object, by default False.
    return_fig : bool, optional
        If True, return the figure and axes objects, by default False.
    """
    # Make sure the kwargs are lists too
    hist_kwargs = make_list(hist_kwargs, len(data_list))
    err_kwargs = make_list(err_kwargs, len(data_list))

    # Create the figure with the many sub axes
    n_features = len(col_labels)
    fig, axes = plt.subplots(
        n_features,
        n_features,
        figsize=((2 * n_features + 3) * fig_scale, (2 * n_features + 1) * fig_scale),
        gridspec_kw={"wspace": 0.04, "hspace": 0.04},
    )

    # Define the binning as auto or not
    all_bins = []
    for n in range(n_features):
        if bins is None or bins[n] == "auto":
            all_bins.append(quantile_bins(data_list[0][:, n], bins=n_bins))
        else:
            all_bins.append(np.array(bins[n]))

    # Cycle through the rows and columns and set the axis labels
    for row in range(n_features):
        axes[0, 0].set_ylabel("A.U.", loc="top")
        if row != 0:
            axes[row, 0].set_ylabel(col_labels[row])
        for column in range(n_features):
            axes[-1, column].set_xlabel(col_labels[column])
            if column != 0:
                axes[row, column].set_yticklabels([])

            # Remove all ticks
            if row != n_features - 1:
                axes[row, column].tick_params(
                    axis="x", which="both", direction="in", labelbottom=False
                )
            if row == column == 0:
                axes[row, column].yaxis.set_ticklabels([])
            elif column > 0:
                axes[row, column].tick_params(
                    axis="y", which="both", direction="in", labelbottom=False
                )

            # For the diagonals they become histograms
            # Bins are based on the first datapoint in the list
            if row == column:
                b = all_bins[column]
                for i, d in enumerate(data_list):
                    add_hist(
                        axes[row, column],
                        d[:, row],
                        bins=b,
                        hist_kwargs=hist_kwargs[i],
                        err_kwargs=err_kwargs[i],
                        do_err=do_err,
                        do_norm=do_norm,
                    )
                    axes[row, column].set_xlim(b[0], b[-1])

            # If we are in the lower triange  fill using a contour plot
            elif row > column:
                x_bounds = np.quantile(data_list[0][:, column], [0.001, 0.999])
                y_bounds = np.quantile(data_list[0][:, row], [0.001, 0.999])
                for i, d in enumerate(data_list):
                    color = None
                    if hist_kwargs[i] is not None and "color" in hist_kwargs[i]:
                        color = hist_kwargs[i]["color"]
                    sns.kdeplot(
                        x=d[:, column],
                        y=d[:, row],
                        ax=axes[row, column],
                        alpha=0.4,
                        levels=levels,
                        color=color,
                        fill=True,
                        clip=[x_bounds, y_bounds],
                        gridsize=n_kde_points,
                    )
                    axes[row, column].set_xlim(x_bounds)
                    axes[row, column].set_ylim(y_bounds)

            # If we are in the upper triangle we set visibility off
            else:
                axes[row, column].set_visible(False)

    # Create some invisible lines which will be part of the legend
    for i in range(len(data_list)):
        color = None
        if hist_kwargs[i] is not None and "color" in hist_kwargs[i]:
            color = hist_kwargs[i]["color"]
        axes[row, column].plot([], [], label=data_labels[i], color=color)
    fig.legend(**(legend_kwargs or {"loc": "upper center", "fontsize": 20}))

    # Save the file
    if path is not None:
        fig.savefig(path)

    # Return a rendered image, or the matplotlib figure, or close
    if return_img:
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close(fig)
        return img
    if return_fig:
        return fig
    plt.close(fig)
    return None


def make_list(x: list | np.ndarray, length: int = 1) -> list:
    """Turn an input into a list of itself repeated length times."""
    if not isinstance(x, list):
        x = length * [x]
    return x


def plot_multi_hists(
    data_list: list | np.ndarray,
    data_labels: list | str,
    col_labels: list | str,
    path: Path | str | None = None,
    bins: list | str | partial = "auto",
    scale_factors: list | None = None,
    do_err: bool = False,
    do_norm: bool = False,
    logy: bool = False,
    y_label: str | None = None,
    ylims: list | tuple | None = None,
    ypad: float = 1.5,
    rat_ylim: tuple | None = (0, 2),
    rat_label: str | None = None,
    fig_height: int = 5,
    do_legend: bool = True,
    ignore_nans: bool = False,
    hist_kwargs: list | None = None,
    err_kwargs: list | None = None,
    legend_kwargs: list | None = None,
    extra_text: list | None = None,
    incl_overflow: bool = True,
    incl_underflow: bool = True,
    do_ratio_to_first: bool = False,
    axis_callbacks: list[Callable] | None = None,
    return_fig: bool = False,
    return_img: bool = False,
) -> plt.Figure | None:
    """Plot multiple histograms given a list of 2D tensors/arrays.

    - Performs the histogramming here
    - Each column the arrays will be a seperate axis
    - Matching columns in each array will be superimposed on the same axis
    - If the tensor being passed is 3D it will average them and combine the uncertainty

    Parameters
    ----------
    data_list : list | np.ndarray
        List of data arrays to be plotted. Each column will be a different axis.
    data_labels : list
        List of labels for the data. Will appear in the legend.
    col_labels : list
        List of column labels for the data. Will appear on the x-axis.
    path : Path | str, optional
        Path where to save the figure, by default None.
    bins : list | str | partial, optional
        List of bin edges, or a string to determine the binning, by default "auto".
    scale_factors : list | None, optional
        List of scaling factors to apply to each element in the data_list
        By default None.
    do_err : bool, optional
        If True, add error bars to the histogram using statistical uncertainty,
        by default False.
    do_norm : bool, optional
        If True, normalize each histogram, by default False.
    logy : bool, optional
        If True, set the y-axis to a log scale, by default False.
    y_label : str | None, optional
        Label for the y-axis, by default None.
    ylims : list | tuple | None, optional
        Tuple of the y-axis limits, by default None.
    ypad : float, optional
        Padding for the y-axis limits, by default 1.5.
    rat_ylim : tuple | None, optional
        Tuple of the y-axis limits for the ratio plot, by default (0, 2).
    rat_label : str | None, optional
        Label for the y-axis of the ratio plot.
        If None it will default to `ratio to data_labels[0]`,
        by default None.
    fig_height : int, optional
        Height of the figure, by default 5.
        Width is determined by the number of axes.
    do_legend : bool, optional
        If True, add a legend to the plot, by default True.
    ignore_nans : bool, optional
        If True, ignore NaNs in the data, by default False.
        If False and NaNs are present, it will raise an error.
    hist_kwargs : list | None, optional
        List of dictionaries with keyword arguments for the plotting function,
        Will be passed to plt.stairs, by default None.
    err_kwargs : list | None, optional
        List of dictionaries with keyword arguments for the error function,
        Will be passed to plt.stairs, by default None.
    legend_kwargs : list | None, optional
        List of dictionaries with keyword arguments for the legend function,
        Will be passed to plt.legend, by default None.
    extra_text : list | None, optional
        List of strings to add to the plot, by default None.
        Will be passed to plt.text.
    incl_overflow : bool, optional
        If True, final bin includes the overflow in the histogram, by default True.
    incl_underflow : bool, optional
        If True, first bin includes the underflow in the histogram, by default True.
    do_ratio_to_first : bool, optional
        If True, add a ratio plot to the bottom of the figure, by default False.
    axis_callbacks : list[Callable] | None, optional
        List of functions to apply to each axis, by default None.
    return_fig : bool, optional
        If True, return the figure and axes objects, by default False.
    return_img
        If True, return the image as a PIL.Image object, by default False.
    """
    # Deep copy the bins as they are mutable
    bins = deepcopy(bins)

    # Make the main arguments lists for generlity
    data_list = make_list(data_list)
    data_labels = make_list(data_labels)
    col_labels = make_list(col_labels)

    # Check the number of histograms to plot and the number of axes
    n_data = len(data_list)
    n_axis = data_list[0].shape[-1]

    # Inputs that depend on the number of data sources
    scale_factors = make_list(scale_factors, n_data)
    hist_kwargs = make_list(hist_kwargs, n_data)
    err_kwargs = make_list(err_kwargs, n_data)

    # Inputs that depend on the number of axes/variables
    bins = make_list(bins, n_axis)
    extra_text = make_list(extra_text, n_axis)
    legend_kwargs = make_list(legend_kwargs, n_axis)
    ylims = make_list(ylims, n_axis)
    axis_callbacks = make_list(axis_callbacks, n_axis)

    # Check all the lengths are correct
    assert len(data_labels) == n_data
    assert len(scale_factors) == n_data
    assert len(hist_kwargs) == n_data
    assert len(err_kwargs) == n_data
    assert len(col_labels) == n_axis
    assert len(bins) == n_axis
    assert len(extra_text) == n_axis
    assert len(legend_kwargs) == n_axis
    assert len(ylims) == n_axis
    assert len(axis_callbacks) == n_axis

    # Cycle through each data entries and make sure they are at least 2D
    for data_idx in range(n_data):
        if data_list[data_idx].ndim < 2:
            data_list[data_idx] = np.expand_dims(data_list[data_idx], -1)

    # Make sure the there are not too many subplots
    if n_axis > 20:
        raise RuntimeError("You are asking to create more than 20 subplots!")

    # Create the figure and axes lists
    dims = np.array([1, n_axis])  # Subplot is (n_rows, n_columns)
    size = np.array([n_axis, 1.0])  # Size is (width, height)
    if do_ratio_to_first:
        dims *= np.array([2, 1])  # Double the number of rows
        size *= np.array([1, 1.2])  # Increase the height
    fig, axes = plt.subplots(
        *dims,
        figsize=tuple(fig_height * size),
        gridspec_kw={"height_ratios": [3, 1] if do_ratio_to_first else {1}},
        squeeze=False,
    )

    # Cycle through each axis and determine the bins that should be used
    # Automatic/Interger bins are replaced using the first item in the data list
    for ax_idx in range(n_axis):
        ax_bins = bins[ax_idx]
        if isinstance(ax_bins, partial):
            ax_bins = ax_bins()

        # The data for this axis
        data = data_list[0][:, ax_idx]
        if ignore_nans and np.isnan(data).any():
            data = data[~np.isnan(data)]

        # If the axis bins was specified to be 'auto' or another numpy string
        if isinstance(ax_bins, str):
            unq = np.unique(data)
            n_unique = len(unq)

            # If the number of datapoints is less than 10 then use even spacing
            if 1 < n_unique < 10:
                ax_bins = (unq[1:] + unq[:-1]) / 2  # Use midpoints, add final, initial
                ax_bins = np.append(ax_bins, unq.max() + unq.max() - ax_bins[-1])
                ax_bins = np.insert(ax_bins, 0, unq.min() + unq.min() - ax_bins[0])

            elif ax_bins == "quant":
                ax_bins = quantile_bins(data)

        # Numpy function to get the bin edges, catches all other cases (int, etc)
        ax_bins = np.histogram_bin_edges(data, bins=ax_bins)

        # Replace the element in the array with the edges
        bins[ax_idx] = ax_bins

    # Cycle through each of the axes
    for ax_idx in range(n_axis):
        # Get the bins for this axis
        ax_bins = bins[ax_idx]

        # Cycle through each of the data arrays
        for data_idx in range(n_data):
            # Get the data to plot (make a copy to avoid changing the original)
            data = np.copy(data_list[data_idx][..., ax_idx]).squeeze()

            # Check for NaNs in the data
            if ignore_nans and np.isnan(data).any():
                data = data[~np.isnan(data)]

            # Clip to get overflow and underflow
            if incl_overflow:
                data = np.minimum(data, ax_bins[-1])
            if incl_underflow:
                data = np.maximum(data, ax_bins[0])

            # If the data is still a 2D tensor treat it as a collection of histograms
            if data.ndim > 1:
                h = [
                    np.histogram(data[:, dim], ax_bins)[0]
                    for dim in range(data.shape[-1])
                ]

                # Nominal and err is based on chi2 of same value with mult measurements
                hist = 1 / np.mean(1 / np.array(h), axis=0)
                hist_err = np.sqrt(1 / np.sum(1 / np.array(h), axis=0))

            # Otherwise just calculate a single histogram with statistical err
            else:
                hist, _ = np.histogram(data, ax_bins)
                hist_err = np.sqrt(hist)

            # Manually do the density so that the error can be scaled appropriately
            if do_norm:
                divisor = np.array(np.diff(ax_bins), float) / hist.sum()
                hist = hist * divisor
                hist_err = hist_err * divisor

            # Apply the scale factors
            if scale_factors[data_idx] is not None:
                hist *= scale_factors
                hist_err *= scale_factors

            # Save the first histogram for the ratio plots
            if data_idx == 0:
                denom_hist = hist

            # Get the additional keyword arguments for drawing the histograms
            if hist_kwargs[data_idx] is not None and bool(hist_kwargs[data_idx]):
                h_kwargs = deepcopy(hist_kwargs[data_idx])
            else:
                h_kwargs = {}  # Use matplotlib defaults

            # Use the stair function to plot the histograms with the kwargs
            line = axes[0, ax_idx].stairs(
                hist, ax_bins, label=data_labels[data_idx], **h_kwargs
            )

            # Include the uncertainty in the plots as a shaded region
            if do_err:
                # Get arguments for drawing the error plots
                if err_kwargs[data_idx] is not None and bool(err_kwargs[data_idx]):
                    e_kwargs = deepcopy(err_kwargs[data_idx])
                else:  # By default use the same colour as the line
                    e_kwargs = {"color": line.get_ec(), "alpha": 0.2, "fill": True}

                # Use stairs with baseline to plot the errors
                axes[0, ax_idx].stairs(
                    hist + hist_err, ax_bins, baseline=hist - hist_err, **e_kwargs
                )

            # Add a ratio plot
            if do_ratio_to_first:
                # Ratio kwargs are the same as hist, if not defined make same color
                ratio_kwargs = deepcopy(h_kwargs) or {
                    "color": line.get_color(),
                    "linestyle": line.get_linestyle(),
                }
                ratio_kwargs["fill"] = False  # Never fill a ratio plot

                # Calculate the new ratio values with their errors
                rat_hist = hist / denom_hist
                rat_err = hist_err / denom_hist

                # Plot the ratios
                axes[1, ax_idx].stairs(rat_hist, ax_bins, **ratio_kwargs)

                # Use the same error kwargs as the histogram
                if do_err:
                    axes[1, ax_idx].stairs(
                        rat_hist + rat_err,
                        ax_bins,
                        baseline=rat_hist - rat_err,
                        **e_kwargs,
                    )

                # Add arrows for values outside the ratio limits
                if rat_ylim is not None:
                    mid_bins = (ax_bins[1:] + ax_bins[:-1]) / 2
                    if not isinstance(rat_ylim, tuple):
                        rat_ylim = tuple(*rat_ylim)
                    ymin, ymax = rat_ylim  # Convert to tuple incase list
                    arrow_height = 0.02 * (ymax - ymin)

                    # Values above the limits
                    mask_up = rat_hist >= ymax
                    up_vals = mid_bins[mask_up]
                    axes[1, ax_idx].arrow(
                        x=up_vals,
                        y=ymax - arrow_height - 0.01,
                        dx=0,
                        dy=arrow_height,
                        color=line.get_color(),
                        width=arrow_height / 2,
                    )

                    # Values below the limits
                    mask_down = rat_hist <= ymin
                    down_vals = mid_bins[mask_down]
                    axes[1, ax_idx].arrow(
                        x=down_vals,
                        y=ymin + 0.01,
                        dx=0,
                        dy=arrow_height,
                        color=line.get_color(),
                        width=arrow_height / 2,
                    )

    # Cycle again through each axis and apply editing
    for ax_idx in range(n_axis):
        ax_bins = bins[ax_idx]

        # Set the x axis limits to the bin edges
        axes[0, ax_idx].set_xlim(ax_bins[0], ax_bins[-1])

        # X-axis
        # If using a ratio plot, remove the ticks from the top plot
        if do_ratio_to_first:
            axes[0, ax_idx].set_xticklabels([])
            axes[1, ax_idx].set_xlabel(col_labels[ax_idx])
            axes[1, ax_idx].set_xlim(ax_bins[0], ax_bins[-1])

        # Otherwise set the x axis label
        else:
            axes[0, ax_idx].set_xlabel(col_labels[ax_idx])

        # Y-axis
        if logy:
            axes[0, ax_idx].set_yscale("log")

        # Set the y axis limits
        if ylims[ax_idx] is not None:
            axes[0, ax_idx].set_ylim(*ylims[ax_idx])

        # Otherwise use predefined padding methods as default is not good enough
        else:
            _, ylim2 = axes[0, ax_idx].get_ylim()
            if logy:
                axes[0, ax_idx].set_ylim(top=np.exp(np.log(ylim2) + ypad))
            else:
                axes[0, ax_idx].set_ylim(top=ylim2 * ypad)
        if y_label is not None:
            axes[0, ax_idx].set_ylabel(y_label)
        elif do_norm:
            axes[0, ax_idx].set_ylabel("Normalised Entries")
        else:
            axes[0, ax_idx].set_ylabel("Entries")

        if do_ratio_to_first:
            # Ratio Y-axis
            if rat_ylim is not None:
                axes[1, ax_idx].set_ylim(rat_ylim)
            if rat_label is not None:
                axes[1, ax_idx].set_ylabel(rat_label)
            else:
                axes[1, ax_idx].set_ylabel(f"Ratio to {data_labels[0]}")

            # Ratio X-line:
            axes[1, ax_idx].hlines(
                1, *axes[1, ax_idx].get_xlim(), colors="k", zorder=-9999
            )

        # Add extra text to main plot
        if extra_text[ax_idx] is not None:
            axes[0, ax_idx].text(**extra_text[ax_idx])

        # Legend
        if do_legend:
            lk = legend_kwargs[ax_idx] or {}
            axes[0, ax_idx].legend(**lk)

        # Any final callbacks to execute on the axis
        if axis_callbacks[ax_idx] is not None:
            axis_callbacks[ax_idx](fig, axes[0, ax_idx])

    # Final figure layout
    fig.tight_layout()
    if do_ratio_to_first:
        fig.subplots_adjust(hspace=0.08)  # Looks neater for ratio plots

    # Save the file
    if path is not None:
        fig.savefig(path)

    # Return a rendered image, or the matplotlib figure, or close
    if return_img:
        img = render_image(fig)
        plt.close(fig)
        return img
    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_2d_hists(path, hist_list, hist_labels, ax_labels, bins):
    """Given a list of 2D histograms, plot them side by side as imshows."""
    # Calculate the axis limits from the bins
    limits = (min(bins[0]), max(bins[0]), min(bins[1]), max(bins[1]))
    mid_bins = [(b[1:] + b[:-1]) / 2 for b in bins]

    # Create the subplots
    fig, axes = plt.subplots(1, len(hist_list), figsize=(8, 4))

    # For each histogram to be plotted
    for i in range(len(hist_list)):
        axes[i].set_xlabel(ax_labels[0])
        axes[i].set_title(hist_labels[i])
        axes[i].imshow(
            hist_list[i], cmap="viridis", origin="lower", extent=limits, norm=LogNorm()
        )
        axes[i].contour(*mid_bins, np.log(hist_list[i] + 1e-4), colors="k", levels=10)

    axes[0].set_ylabel(ax_labels[1])
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
