import numpy as np
import plotly 
import plotly.graph_objs as go
from scipy.stats import norm, gaussian_kde

from .utils import snorm_qq

def hist_residuals(y_pred_mean, y_pred_var, y_test, title = '', showlegend = True):
    """
    Create histogram of residuals
    Input: 
    y_pred_mean - prediction mean
    y_pred_var - prediction variance
    y_test - true y value
    title - (optional) 
    Output:
    fig - plotly figure
    """

    # defualt color etc.. 
    clr = 'rgb(105, 144, 193)'

    # Calculate residuals
    y_pred_std = np.power(y_pred_var, 0.5)
    residuals_y = (y_pred_mean - y_test)/y_pred_std # Standardized residuals
    
    # Histogram
    hist = go.Histogram(x=residuals_y, histnorm='probability density', marker_color=clr, opacity = 0.5,
                       name = 'residuals')

    # Standard normal density
    x_min, x_max = residuals_y.min(), residuals_y.max()
    x = np.linspace(x_min, x_max, 100)
    normal_dens = go.Scatter(x = x, y = norm.pdf(x), mode = 'lines', 
                             name = 'standard normal density', line = dict(color = 'black', width = 1))

    # kde
    kde = gaussian_kde(residuals_y)
    kde_dens = go.Scatter(x = x, y = kde(x), mode = 'lines', 
                             name = 'residuals kde', line = dict(color = clr, width = 1))

    # Layout
    layout = go.Layout(
            title=title,
            showlegend=showlegend,
            autosize=False,
            width=700,
            height=600,
            xaxis=dict(
                title='Standardised error',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            ),
            yaxis=dict(
                title='Density',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )



    data = [hist, kde_dens, normal_dens]
    fig = go.Figure(data=data, layout=layout)
    
    return fig

def qq_residuals(y_pred_mean, y_pred_var, y_test, title = '', showlegend = True):
    """
    Create QQ plot
    Input: 
    y_pred_mean - prediction mean
    y_pred_var - prediction variance
    y_test - true y value
    title - (optional) 
    Output:
    fig - plotly figure
    """

    # Calculate residuals
    y_pred_std = np.power(y_pred_var, 0.5)
    residuals_y = (y_pred_mean - y_test)/y_pred_std # Standardized residuals

    # Calculate QQ data
    q_sample, q_snorm, q_snorm_upper, q_snorm_lower = snorm_qq(residuals_y)

    qq_scatter = go.Scatter(
        x = q_snorm,
        y = q_sample,
        mode = 'markers',
        marker = dict(
            size = 6,
            color='rgb(105, 144, 193)'
        ),
        name = 'Data'
    )

    qq_upper = go.Scatter(
        x = q_snorm_upper,
        y = q_sample,
        mode = 'lines',
        line = dict(
            color='rgb(150, 150, 150)',
            dash = 'dot'
        ),
        name = '95% confidence band',
        legendgroup = 'conf'
    )

    qq_lower = go.Scatter(
        x = q_snorm_lower,
        y = q_sample,
        mode = 'lines',
        line = dict(
            color='rgb(150, 150, 150)',
            dash = 'dot'
        ),
        name = 'lower',
        legendgroup = 'conf',
        showlegend = False
    )

    minval = np.min([q_sample.min(), q_snorm.min()])
    maxval = np.max([q_sample.min(), q_snorm.max()])

    line = go.Scatter(
        x = [minval, maxval],
        y = [minval, maxval],
        mode = 'lines',
        line = dict(
            color='rgb(0, 0, 0)',
            dash = 'dash'
        ),
        name = 'x = y'
    )

    layout = go.Layout(
        title=title,
        showlegend=showlegend,
        autosize=False,
        width=700,
        height=600,
        xaxis=dict(
            title='Standard normal quantiles',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            ),
            range = [q_snorm.min() - 0.2, q_snorm.max() + 0.2]
        ),
        yaxis=dict(
            title='Sample quantiles',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    data = [qq_scatter, qq_upper, qq_lower, line]
    fig = go.Figure(data=data, layout=layout)
    
    return fig

def add_traces_to_fig(fig, trace_list):
    """ 
    Add traces to plotly figure 
    
    fig - plotly figure
    trace_list - list of new traces
    """
    
    data = list(fig.data) + trace_list
    layout = fig.layout
    
    return go.Figure(data = data, layout = layout)
	
def pred_vs_error(y_pred_mean, y_pred_var, y_test, title = '', showlegend = True):
    """
    Plot prediction with error bars as a function of true value
    Input: 
    y_pred_mean - prediction mean
    y_pred_var - prediction variance
    y_test - true y value
    Output:
    fig - plotly figure
    """

    # Plot min/max
    minval = np.min([y_pred_mean.min(), y_test.min()])
    maxval = np.max([y_pred_mean.min(), y_test.max()])

    # Prediction standard deviation
    y_pred_std = np.power(y_pred_var, 0.5)

    # Predictions
    pred = go.Scatter(
        x = y_test,
        y = y_pred_mean,
        mode = 'markers',
        marker = dict(
            size = 6,
            color='rgb(105, 144, 193)'
        ),
        name = 'Prediction'
    )

    # Predictions with error bars
    num_std = 1.959963984540054
    pred_bars = go.Scatter(
        x = y_test,
        y = y_pred_mean,
        mode = 'markers',
        marker = dict(
            size = 6,
            color='rgb(105, 144, 193)'
        ),
        error_y = dict(
                type='data',
                array=y_pred_std*num_std,
                visible=True,
                color='rgb(105, 144, 193)'
        ),
        name = '95% intervals'
    )

    line = go.Scatter(
        x = [minval, maxval],
        y = [minval, maxval],
        mode = 'lines',
        line = dict(
            color='rgb(0, 0, 0)',
            dash = 'dash'
        ),
        name = 'x = y'
    )

    layout = go.Layout(
        title=title,
        showlegend=showlegend,
        autosize=False,
        width=700,
        height=600,
        xaxis=dict(
            title='True value',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Predicted value',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    data = [pred_bars, pred, line]
    fig = go.Figure(data=data, layout=layout)
    
    return fig

def pred_vs_error_perc(y_pred_mean, y_pred_perc_lower, y_pred_perc_upper, y_test, conf_interval, title = '', showlegend = True):
    """
    Plot prediction with error bars as a function of true value -- using percentile data
    
    Input: 
    y_pred_mean - prediction mean
    y_pred_perc_lower, y_pred_perc_upper - prediction percentiles
    y_test - true y value
    conf_interval - % covered by lower/upper percentiles (upper - lower)
    Output:
    fig - plotly figure
    """

    # Plot min/max
    minval = np.min([y_pred_mean.min(), y_test.min()])
    maxval = np.max([y_pred_mean.min(), y_test.max()])

    # Predictions
    pred = go.Scatter(
        x = y_test,
        y = y_pred_mean,
        mode = 'markers',
        marker = dict(
            size = 6,
            color='rgb(105, 144, 193)'
        ),
        name = 'Prediction'
    )

    # Predictions with error bars
    pred_bars = go.Scatter(
        x = y_test,
        y = (y_pred_perc_upper + y_pred_perc_lower)/2,
        mode = 'markers',
        marker = dict(
            size = 0,
            color='rgb(105, 144, 193)',
            opacity = 0
        ),
        error_y = dict(
                type='data',
                array=(y_pred_perc_upper - y_pred_perc_lower)/2,
                visible=True,
                color='rgb(105, 144, 193)'
        ),
        name = '{}% intervals'.format(conf_interval)
    )

    line = go.Scatter(
        x = [minval, maxval],
        y = [minval, maxval],
        mode = 'lines',
        line = dict(
            color='rgb(0, 0, 0)',
            dash = 'dash'
        ),
        name = 'x = y'
    )

    layout = go.Layout(
        title=title,
        showlegend=showlegend,
        autosize=False,
        width=700,
        height=600,
        xaxis=dict(
            title='True value',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Predicted value',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    data = [pred_bars, pred, line]
    fig = go.Figure(data=data, layout=layout)
    
    return fig

def error_scatter(x, errors, title = '', x_label = 'x', y_label = 'Standardized errors', showlegend = True):
    """
    Error scatter plot with 95% interval
    """

    errors = go.Scatter(
        x = x,
        y = errors,
        mode = 'markers',
        marker = dict(
            size = 6,
            color='rgb(105, 144, 193)'
        ),
        name = y_label
    )


    num_std = 1.959963984540054 # For 95% interval
    min_x = x.min()
    max_x = x.max()
    margin = (max_x - min_x)*0.025 # For plotting
    x_line = [min_x - margin, max_x + margin]

    line_mid = go.Scatter(
        x = x_line,
        y = [0, 0],
        mode = 'lines',
        line = dict(
            color='rgb(0, 0, 0)',
            dash = 'dash'
        ),
        name = '',
        showlegend = False
    )

    line_upper = go.Scatter(
        x = x_line,
        y = [num_std, num_std],
        mode = 'lines',
        line = dict(
            color='rgb(150, 150, 150)',
            dash = 'dot'
        ),
        name = '95% interval',
        legendgroup = 'conf'
    )

    line_lower = go.Scatter(
        x = x_line,
        y = [-num_std, -num_std],
        mode = 'lines',
        line = dict(
            color='rgb(150, 150, 150)',
            dash = 'dot'
        ),
        name = '95% interval',
        legendgroup = 'conf',
        showlegend = False
    )

    layout = go.Layout(
        title=title,
        showlegend = showlegend,
        autosize=False,
        width=700,
        height=600,
        xaxis=dict(
            title=x_label,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            ),
            range = [min_x - margin, max_x + margin]
        ),
        yaxis=dict(
            title=y_label,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    data = [errors, line_mid, line_upper, line_lower]
    fig = go.Figure(data=data, layout=layout)
    
    return fig

def gp_diagnostics(data, y_name, plot_labels = {}, subplots = True):
    """
    Returns list of plotly figures for GP diagnostics
    Inputs:

    data - Pandas dataframe with all input and output data. 
           Example: columns = x1 x2 x3 x4 y_true y_mean y_var

    y_name - Name of output variable. The dataframe 'data' must contain columns
             named y_name+'_true', y_name+'_mean' and y_name+'_var'.
             All other columns are assumed to be input variables

    plot_labels - (optional) Dictionary for renaming variables in plots
                  Example: {'x1': 'variable 1', 'y': 'output'}

    subplots - (optional) The figures will be put in subplots for a more compact view
    """
    
    # Name of inputs and outputs
    outputnames = [y_name + sub for sub in ['_true', '_mean', '_var']]
    inputnames = [name for name in data.columns if not name in outputnames]
    
    # Check that output is specified correctly 
    if not all(name in data.columns for name in outputnames):
        print('Error: data must contain the following columns for the output named ' + y_name)
        print(outputnames)
        return 0
    
    # Extract data from dataframe
    y_pred_mean = data[y_name + '_mean']
    y_pred_var = data[y_name + '_var']
    y_test = data[y_name + '_true']
    y_pred_std = np.power(y_pred_var, 0.5)
    residuals_y = (y_pred_mean - y_test)/y_pred_std # Standardized residuals
    
    # Create complete dict for all plot axis labels
    dict_varnames = plot_labels
    for name in data.columns:
        if not name in dict_varnames.keys():
            dict_varnames[name] = name
            
    if not y_name in dict_varnames.keys():
        dict_varnames[y_name] = y_name
    
    # QQ plot
    title = 'Normal QQ plot of standardised errors with 95% confidence band'
    fig_qq = qq_residuals(y_pred_mean, y_pred_var, y_test, title, showlegend=False)
    
    # Prediction vs test
    title = 'Prediction vs test with 95% intervals'
    fig_pred_vs_err = pred_vs_error(y_pred_mean, y_pred_var, y_test, title = title, showlegend=False)
    
    # Scatter plots of standardised errors
    x = data[y_name + '_mean']
    x_label = 'GP mean E[' + dict_varnames[y_name] + ']'
    title = 'Standardised errors as a function of GP mean'
    y_label = 'Standardised errors'
    figs_errorscatter = [error_scatter(x, residuals_y, title, x_label, y_label, showlegend = False)]

    x = data[y_name + '_var']
    x_label = 'GP variance Var[' + dict_varnames[y_name] + ']'
    title = 'Standardised errors as a function of GP variance'
    y_label = 'Standardised errors'
    figs_errorscatter.append(error_scatter(x, residuals_y, title, x_label, y_label, showlegend = False))
    
    for name in inputnames:
        x = data[name]
        x_label = dict_varnames[name]
        title = 'Standardised errors as a function of ' + x_label
        y_label = 'Standardised errors'
        figs_errorscatter.append(error_scatter(x, residuals_y, title, x_label, y_label, showlegend = False))

    if not subplots:
        # Return list of all plots
        return [fig_qq, fig_pred_vs_err] + figs_errorscatter
    
    else:
        # 1. Figure with QQ and pred vs error
        fig1 = plotly.subplots.make_subplots(rows=1, cols=2, 
                                 subplot_titles=('Prediction vs test', 'Standardised errors QQ'), print_grid=False)

        for trace in fig_qq['data']:
            fig1.append_trace(trace, 1, 2)

        for trace in fig_pred_vs_err['data']:
            fig1.append_trace(trace, 1, 1)

        fig1['layout']['xaxis2'].update(fig_qq['layout']['xaxis'])
        fig1['layout']['yaxis2'].update(fig_qq['layout']['yaxis'])

        fig1['layout']['xaxis1'].update(fig_pred_vs_err['layout']['xaxis'])
        fig1['layout']['yaxis1'].update(fig_pred_vs_err['layout']['yaxis'])

        fig1['layout'].update(showlegend=False)
        
        # 2. Pivoted chol....
        
        
        # 3. Standardised error scatter plots
        numcols = 3
        numplots = len(figs_errorscatter)
        numrows = int(np.ceil(numplots/numcols))

        fig3 = plotly.subplots.make_subplots(rows=numrows, cols=numcols, print_grid=False)

        index = -1
        for i in range(numcols):
            for j in range(numrows):
                index += 1 
                if index < numplots:
                    for trace in figs_errorscatter[index]['data']:
                        fig3.append_trace(trace, j+1, i+1)


                    fig3['layout']['xaxis' + str(index + 1)].update(title = figs_errorscatter[index]['layout']['xaxis']['title'])


        fig3['layout'].update(showlegend=False, title = 'Standardised errors')
        fig3['layout'].update(height=330*numrows, width=950)

        return [fig1, fig3]
        