"""Plotly-based functions for visualizing residuals, predictions, QQ plots, histograms, etc.

Used to aid in GP diagnostics.
"""

from __future__ import annotations

__all__ = [
    "error_scatter",
    "gp_diagnostics",
    "hist_residuals",
    "pred_vs_error",
    "pred_vs_error_perc",
    "qq_residuals",
]

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, norm

if TYPE_CHECKING:
    import pandas as pd

from gp_diagnostics.utils.stats import snorm_qq


def hist_residuals(
    y_pred_mean: npt.NDArray[np.float64],
    y_pred_var: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
    *,
    title: str = "",
    showlegend: bool = True,
) -> plotly.graph_objs.Figure:
    """Creates a Plotly histogram of standardized residuals with normal & KDE overlays.

    Args:
        y_pred_mean: Predicted means, shape (n_samples,).
        y_pred_var: Predicted variances, shape (n_samples,).
        y_test: True target values, shape (n_samples,).
        title: Plot title.
        showlegend: Whether to show the legend.

    Returns:
        A Plotly figure with a histogram of residuals.
    """
    clr = "rgb(105, 144, 193)"
    y_pred_std = np.sqrt(y_pred_var)
    residuals_y = (y_pred_mean - y_test) / y_pred_std

    hist = go.Histogram(
        x=residuals_y,
        histnorm="probability density",
        marker_color=clr,
        opacity=0.5,
        name="residuals",
    )

    x_min, x_max = float(residuals_y.min()), float(residuals_y.max())
    x_points = np.linspace(x_min, x_max, 100)

    normal_dens = go.Scatter(
        x=x_points,
        y=norm.pdf(x_points),
        mode="lines",
        name="standard normal density",
        line={"color": "black", "width": 1},
    )
    kde = gaussian_kde(residuals_y)
    kde_dens = go.Scatter(
        x=x_points,
        y=kde(x_points),
        mode="lines",
        name="residuals kde",
        line={"color": clr, "width": 1},
    )

    layout = go.Layout(
        title=title,
        showlegend=showlegend,
        autosize=False,
        width=700,
        height=600,
        xaxis={"title": "Standardised error"},
        yaxis={"title": "Density"},
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    data = [hist, kde_dens, normal_dens]
    return go.Figure(data=data, layout=layout)


def qq_residuals(
    y_pred_mean: npt.NDArray[np.float64],
    y_pred_var: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
    *,
    title: str = "",
    showlegend: bool = True,
) -> plotly.graph_objs.Figure:
    """Creates a Plotly QQ-plot of standardized residuals vs. standard normal quantiles.

    Args:
        y_pred_mean: Predicted means, shape (n_samples,).
        y_pred_var: Predicted variances, shape (n_samples,).
        y_test: True target values, shape (n_samples,).
        title: Plot title.
        showlegend: Whether to show the legend.

    Returns:
        A Plotly Figure containing the QQ-plot.
    """
    y_pred_std = np.sqrt(y_pred_var)
    residuals_y = (y_pred_mean - y_test) / y_pred_std

    q_sample, q_snorm, q_snorm_upper, q_snorm_lower = snorm_qq(residuals_y)

    qq_scatter = go.Scatter(
        x=q_snorm,
        y=q_sample,
        mode="markers",
        marker={"size": 6, "color": "rgb(105, 144, 193)"},
        name="Data",
    )
    qq_upper = go.Scatter(
        x=q_snorm_upper,
        y=q_sample,
        mode="lines",
        line={"color": "rgb(150, 150, 150)", "dash": "dot"},
        name="95% confidence band",
        legendgroup="conf",
    )
    qq_lower = go.Scatter(
        x=q_snorm_lower,
        y=q_sample,
        mode="lines",
        line={"color": "rgb(150, 150, 150)", "dash": "dot"},
        name="lower",
        legendgroup="conf",
        showlegend=False,
    )

    minval = float(min(q_snorm.min(), q_sample.min()))
    maxval = float(max(q_snorm.max(), q_sample.max()))

    line = go.Scatter(
        x=[minval, maxval],
        y=[minval, maxval],
        mode="lines",
        line={"color": "rgb(0, 0, 0)", "dash": "dash"},
        name="x = y",
    )

    layout = go.Layout(
        title=title,
        showlegend=showlegend,
        autosize=False,
        width=700,
        height=600,
        xaxis={
            "title": "Standard normal quantiles",
            "range": [q_snorm.min() - 0.2, q_snorm.max() + 0.2],
        },
        yaxis={"title": "Sample quantiles"},
    )

    data = [qq_scatter, qq_upper, qq_lower, line]
    return go.Figure(data=data, layout=layout)


def pred_vs_error(
    y_pred_mean: npt.NDArray[np.float64],
    y_pred_var: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
    *,
    title: str = "",
    showlegend: bool = True,
) -> plotly.graph_objs.Figure:
    """Creates a scatter plot comparing predicted vs. true values, with 95% error bars.

    Args:
        y_pred_mean: Predicted means, shape (n_samples,).
        y_pred_var: Predicted variances, shape (n_samples,).
        y_test: True target values, shape (n_samples,).
        title: Plot title.
        showlegend: Whether to show the legend.

    Returns:
        A Plotly Figure showing predicted vs. actual + 95% intervals.
    """
    minval = float(min(y_pred_mean.min(), y_test.min()))
    maxval = float(max(y_pred_mean.max(), y_test.max()))

    y_pred_std = np.sqrt(y_pred_var)
    num_std = 1.959963984540054

    pred_scatter = go.Scatter(
        x=y_test,
        y=y_pred_mean,
        mode="markers",
        marker={"size": 6, "color": "rgb(105, 144, 193)"},
        name="Prediction",
    )

    pred_bars = go.Scatter(
        x=y_test,
        y=y_pred_mean,
        mode="markers",
        marker={"size": 6, "color": "rgb(105, 144, 193)"},
        error_y={
            "type": "data",
            "array": y_pred_std * num_std,
            "visible": True,
            "color": "rgb(105, 144, 193)",
        },
        name="95% intervals",
    )

    line = go.Scatter(
        x=[minval, maxval],
        y=[minval, maxval],
        mode="lines",
        line={"color": "rgb(0, 0, 0)", "dash": "dash"},
        name="x = y",
    )

    layout = go.Layout(
        title=title,
        showlegend=showlegend,
        autosize=False,
        width=700,
        height=600,
        xaxis={"title": "True value"},
        yaxis={"title": "Predicted value"},
    )

    data = [pred_bars, pred_scatter, line]
    return go.Figure(data=data, layout=layout)


def pred_vs_error_perc(
    y_pred_mean: npt.NDArray[np.float64],
    y_pred_perc_lower: npt.NDArray[np.float64],
    y_pred_perc_upper: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
    conf_interval: float,
    *,
    title: str = "",
    showlegend: bool = True,
) -> plotly.graph_objs.Figure:
    """Creates a scatter plot of predicted vs. true values with percentile error bars.

    Args:
        y_pred_mean: Predicted means, shape (n_samples,).
        y_pred_perc_lower: Lower percentile predictions, shape (n_samples,).
        y_pred_perc_upper: Upper percentile predictions, shape (n_samples,).
        y_test: True target values, shape (n_samples,).
        conf_interval: The percentile coverage (e.g. 95).
        title: Plot title.
        showlegend: Whether to show the legend.

    Returns:
        A Plotly Figure with predicted vs. actual scatter + percentile error bars.
    """
    minval = float(min(y_pred_mean.min(), y_test.min()))
    maxval = float(max(y_pred_mean.max(), y_test.max()))

    pred_scatter = go.Scatter(
        x=y_test,
        y=y_pred_mean,
        mode="markers",
        marker={"size": 6, "color": "rgb(105, 144, 193)"},
        name="Prediction",
    )

    pred_bars = go.Scatter(
        x=y_test,
        y=(y_pred_perc_upper + y_pred_perc_lower) / 2.0,
        mode="markers",
        marker={"size": 0, "color": "rgb(105, 144, 193)", "opacity": 0},
        error_y={
            "type": "data",
            "array": (y_pred_perc_upper - y_pred_perc_lower) / 2.0,
            "visible": True,
            "color": "rgb(105, 144, 193)",
        },
        name=f"{conf_interval}% intervals",
    )

    line = go.Scatter(
        x=[minval, maxval],
        y=[minval, maxval],
        mode="lines",
        line={"color": "rgb(0, 0, 0)", "dash": "dash"},
        name="x = y",
    )

    layout = go.Layout(
        title=title,
        showlegend=showlegend,
        autosize=False,
        width=700,
        height=600,
        xaxis={"title": "True value"},
        yaxis={"title": "Predicted value"},
    )

    data = [pred_bars, pred_scatter, line]
    return go.Figure(data=data, layout=layout)


def error_scatter(
    x: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
    *,
    title: str = "",
    x_label: str = "x",
    y_label: str = "Standardized errors",
    showlegend: bool = True,
) -> plotly.graph_objs.Figure:
    """Creates a scatter plot of standardized errors vs. some variable x, with 95% reference lines.

    Args:
        x: Horizontal-axis variable, shape (n_samples,).
        errors: Standardized error values, shape (n_samples,).
        title: Plot title.
        x_label: Label for x-axis.
        y_label: Label for y-axis.
        showlegend: Whether to display the legend.

    Returns:
        A Plotly Figure of the scatter + reference lines at ±1.96 std dev.
    """
    scatter_points = go.Scatter(
        x=x,
        y=errors,
        mode="markers",
        marker={"size": 6, "color": "rgb(105, 144, 193)"},
        name=y_label,
    )

    num_std = 1.959963984540054
    min_x = float(x.min())
    max_x = float(x.max())
    margin = (max_x - min_x) * 0.025
    x_line = [min_x - margin, max_x + margin]

    line_mid = go.Scatter(
        x=x_line,
        y=[0, 0],
        mode="lines",
        line={"color": "rgb(0, 0, 0)", "dash": "dash"},
        name="",
        showlegend=False,
    )
    line_upper = go.Scatter(
        x=x_line,
        y=[num_std, num_std],
        mode="lines",
        line={"color": "rgb(150, 150, 150)", "dash": "dot"},
        name="95% interval",
        legendgroup="conf",
    )
    line_lower = go.Scatter(
        x=x_line,
        y=[-num_std, -num_std],
        mode="lines",
        line={"color": "rgb(150, 150, 150)", "dash": "dot"},
        name="95% interval",
        legendgroup="conf",
        showlegend=False,
    )

    layout = go.Layout(
        title=title,
        showlegend=showlegend,
        autosize=False,
        width=700,
        height=600,
        xaxis={"title": x_label, "range": [x_line[0], x_line[1]]},
        yaxis={"title": y_label},
    )

    data = [scatter_points, line_mid, line_upper, line_lower]
    return go.Figure(data=data, layout=layout)


def gp_diagnostics(
    data: pd.DataFrame,
    y_name: str,
    *,
    plot_labels: dict[str, str] | None = None,
    subplots: bool = True,
) -> list[plotly.graph_objs.Figure]:
    """Creates a set of Plotly figures for GP diagnostics (QQ, pred vs. true, error scatter).

    Args:
        data: Pandas DataFrame with columns for inputs and outputs. Must contain {y_name}_true, {y_name}_mean, and
              {y_name}_var.
        y_name: Base name of the output variable.
        plot_labels: Optional dict to rename variables in plot titles.
        subplots: If True, returns combined subplots for some figures.

    Returns:
        A list of Plotly figures for GP diagnostics, or subplots if requested.

    Raises:
        ValueError: If required columns for y_name are missing in data.
    """
    if plot_labels is None:
        plot_labels = {}

    outputnames = [f"{y_name}_true", f"{y_name}_mean", f"{y_name}_var"]
    inputnames = [col for col in data.columns if col not in outputnames]

    missing_cols = [col for col in outputnames if col not in data.columns]
    if missing_cols:
        msg = f"DataFrame is missing required columns for '{y_name}'.\nMissing columns: {missing_cols}"
        raise ValueError(msg)

    y_pred_mean = data[f"{y_name}_mean"].to_numpy(dtype=np.float64)
    y_pred_var = data[f"{y_name}_var"].to_numpy(dtype=np.float64)
    y_test = data[f"{y_name}_true"].to_numpy(dtype=np.float64)
    y_pred_std = np.sqrt(y_pred_var)
    residuals_y = (y_pred_mean - y_test) / y_pred_std

    dict_varnames = dict(plot_labels)
    for name in data.columns:
        if name not in dict_varnames:
            dict_varnames[name] = name
    if y_name not in dict_varnames:
        dict_varnames[y_name] = y_name

    # QQ plot
    fig_qq = qq_residuals(y_pred_mean, y_pred_var, y_test, title="Normal QQ of errors", showlegend=False)

    # Prediction vs test
    fig_pred_vs_err = pred_vs_error(y_pred_mean, y_pred_var, y_test, title="Prediction vs test", showlegend=False)

    # Standardised error scatter
    figs_errorscatter = []
    x_vals = y_pred_mean
    x_label = f"GP mean E[{dict_varnames[y_name]}]"
    figs_errorscatter.append(
        error_scatter(
            x_vals,
            residuals_y,
            title="Standardised errors vs mean",
            x_label=x_label,
            y_label="Standardised errors",
            showlegend=False,
        )
    )

    x_vals = y_pred_var
    x_label = f"GP variance Var[{dict_varnames[y_name]}]"
    figs_errorscatter.append(
        error_scatter(
            x_vals,
            residuals_y,
            title="Standardised errors vs variance",
            x_label=x_label,
            y_label="Standardised errors",
            showlegend=False,
        )
    )

    for name in inputnames:
        x_vals = data[name].to_numpy(dtype=np.float64)
        x_label = dict_varnames[name]
        figs_errorscatter.append(
            error_scatter(
                x_vals,
                residuals_y,
                title=f"Standardised errors vs {x_label}",
                x_label=x_label,
                y_label="Standardised errors",
                showlegend=False,
            )
        )

    if not subplots:
        return [fig_qq, fig_pred_vs_err, *figs_errorscatter]

    fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Prediction vs test", "Standardised errors QQ"))
    for trace in fig_pred_vs_err.data:
        fig1.add_trace(trace, row=1, col=1)
    for trace in fig_qq.data:
        fig1.add_trace(trace, row=1, col=2)

    fig1.update_xaxes(fig_pred_vs_err.layout.xaxis, row=1, col=1)
    fig1.update_yaxes(fig_pred_vs_err.layout.yaxis, row=1, col=1)
    fig1.update_xaxes(fig_qq.layout.xaxis, row=1, col=2)
    fig1.update_yaxes(fig_qq.layout.yaxis, row=1, col=2)
    fig1.update_layout(showlegend=False)

    numcols = 3
    numplots = len(figs_errorscatter)
    numrows = (numplots + numcols - 1) // numcols

    fig3 = make_subplots(rows=numrows, cols=numcols)
    idx = 0
    for i in range(numrows):
        for j in range(numcols):
            if idx < numplots:
                for trc in figs_errorscatter[idx].data:
                    fig3.add_trace(trc, row=i + 1, col=j + 1)
                fig3.update_xaxes(
                    title=figs_errorscatter[idx].layout.xaxis.title,
                    row=i + 1,
                    col=j + 1,
                )
            idx += 1

    fig3.update_layout(showlegend=False, title="Standardised errors", height=330 * numrows, width=950)

    return [fig1, fig3]
