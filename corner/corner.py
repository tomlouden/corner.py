# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import logging
import numpy as np
from matplotlib.cm import viridis_r, get_cmap
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter, ListedColormap
from matplotlib.ticker import ScalarFormatter
from scipy import interpolate
from posterior import hpd, mode
import math

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

__all__ = ["corner", "hist2d", "quantile","gen_contours"]


def corner(xs, bins=20, drange=None, weights=None, color="C1",
           smooth=None, smooth1d=None,
           labels=None, label_kwargs=None,
           show_titles=True, title_fmt=".2f", title_kwargs=None,
           truths=None, truth_color="#4682b4",
           scale_hist=False, quantiles=[0.68,0.95], verbose=False, fig=None,
           max_n_ticks=5, top_ticks=False, use_math_text=False, reverse=False,
           hex=True, priors=[], set_lims="prior", use_hpd = True,
           hist_kwargs=None, sig_fig=1, units=None, **hist2d_kwargs):
    """
    Make a *sick* corner plot showing the projections of a data set in a
    multi-dimensional space. kwargs are passed to hist2d() or used for
    `matplotlib` styling.

    Parameters
    ----------
    xs : array_like[nsamples, ndim]
        The samples. This should be a 1- or 2-dimensional array. For a 1-D
        array this results in a simple histogram. For a 2-D array, the zeroth
        axis is the list of samples and the next axis are the dimensions of
        the space.

    bins : int or array_like[ndim,]
        The number of bins to use in histograms, either as a fixed value for
        all dimensions, or as a list of integers for each dimension.

    weights : array_like[nsamples,]
        The weight of each sample. If `None` (default), samples are given
        equal weight.

    color : str
        A ``matplotlib`` style color for all histograms.

    smooth, smooth1d : float
       The standard deviation for Gaussian kernel passed to
       `scipy.ndimage.gaussian_filter` to smooth the 2-D and 1-D histograms
       respectively. If `None` (default), no smoothing is applied.

    labels : iterable (ndim,)
        A list of names for the dimensions. If a ``xs`` is a
        ``pandas.DataFrame``, labels will default to column names.

    label_kwargs : dict
        Any extra keyword arguments to send to the `set_xlabel` and
        `set_ylabel` methods.

    show_titles : bool
        Displays a title above each 1-D histogram showing the 0.5 quantile
        with the upper and lower errors supplied by the quantiles argument.

    title_fmt : string
        The format string for the quantiles given in titles. If you explicitly
        set ``show_titles=True`` and ``title_fmt=None``, the labels will be
        shown as the titles. (default: ``.2f``)

    title_kwargs : dict
        Any extra keyword arguments to send to the `set_title` command.

    drange : iterable (ndim,)
        A list where each element is either a length 2 tuple containing
        lower and upper bounds or a float in drange (0., 1.)
        giving the fraction of samples to include in bounds, e.g.,
        [(0.,10.), (1.,5), 0.999, etc.].
        If a fraction, the bounds are chosen to be equal-tailed.

    truths : iterable (ndim,)
        A list of reference values to indicate on the plots.  Individual
        values can be omitted by using ``None``.

    truth_color : str
        A ``matplotlib`` style color for the ``truths`` makers.

    scale_hist : bool
        Should the 1-D histograms be scaled in such a way that the zero line
        is visible?

    quantiles : iterable
        A list of fractional quantiles to show on the 1-D histograms as
        vertical dashed lines.

    verbose : bool
        If true, print the values of the computed quantiles.

    plot_contours : bool
        Draw contours for dense regions of the plot.

    use_math_text : bool
        If true, then axis tick labels for very large or small exponents will
        be displayed as powers of 10 rather than using `e`.
        
    reverse : bool
        If true, plot the corner plot starting in the upper-right corner instead 
        of the usual bottom-left corner

    hex : bool
        If true, use hexbins instead of square bins

    priors : iterable
        A list of prior histograms to be overplotted on the diagonal axis

    set_lims : str
        Set the axis limits from the "priors" or from the "data"
        If you're running a multinest it makes more sense to set prior
        anything else use data set.

    use_hpd : bool
        use the highest posterior density region when reporting the quantiles
        
    max_n_ticks: int
        Maximum number of ticks to try to use

    top_ticks : bool
        If true, label the top ticks of each axis

    fig : matplotlib.Figure
        Overplot onto the provided figure object.

    hist_kwargs : dict
        Any extra keyword arguments to send to the 1-D histogram plots.

    sig_fig : int
        how many significant figures to report errors and best fitting numbers.

    units : iterable
        A list of the units for each of the parameters

    **hist2d_kwargs
        Any remaining keyword arguments are sent to `corner.hist2d` to generate
        the 2-D histogram plots.

    """
    if quantiles is None:
        quantiles = []
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()

    # Try filling in labels from pandas.DataFrame columns.
    if labels is None:
        try:
            labels = xs.columns
        except AttributeError:
            pass

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"

    # Parse the weight array.
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")

    # Parse the parameter dranges.
    if drange is None:
        if "extents" in hist2d_kwargs:
            logging.warn("Deprecated keyword argument 'extents'. "
                         "Use 'range' instead.")
            drange = hist2d_kwargs.pop("extents")
        else:
            drange = [[x.min(), x.max()] for x in xs]
            # Check for parameters that never change.
            m = np.array([e[0] == e[1] for e in drange], dtype=bool)
            if np.any(m):
                raise ValueError(("It looks like the parameter(s) in "
                                  "column(s) {0} have no dynamic drange. "
                                  "Please provide a `range` argument.")
                                 .format(", ".join(map(
                                     "{0}".format, np.arange(len(m))[m]))))

    else:
        # If any of the extents are percentiles, convert them to dranges.
        # Also make sure it's a normal list.
        drange = list(drange)
        for i, _ in enumerate(drange):
            try:
                emin, emax = drange[i]
            except TypeError:
                q = [0.5 - 0.5*range[i], 0.5 + 0.5*range[i]]
                drange[i] = quantile(xs[i], q, weights=weights)

    if len(drange) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and drange")

    prange = []
    for i in range(0,len(priors[0])):
        p = priors[0][i]
        p = p[np.isfinite(p)]
        prange += [[p.min(),p.max()]]

    # Parse the bin specifications.
    try:
        bins = [int(bins) for _ in drange]
    except TypeError:
        if len(bins) != len(drange):
            raise ValueError("Dimension mismatch between bins and drange")

    # Some magic numbers for pretty axis layout.
    K = len(xs)
    factor = 2.0           # size of one side of one panel
    if reverse:
        lbdim = 0.2 * factor   # size of left/bottom margin
        trdim = 0.5 * factor   # size of top/right margin
    else:
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    if fig is None:
        fig, axes = pl.subplots(K, K, figsize=(dim, dim),aspect="equal")
    else:
        try:
            axes = np.array(fig.axes).reshape((K, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                             "dimensions K={1}".format(len(fig.axes), K))

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    # Set up the default histogram keywords.
    if hist_kwargs is None:
        hist_kwargs = dict()
    hist_kwargs["color"] = hist_kwargs.get("color", color)
    if smooth1d is None:
        hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")

    for i, x in enumerate(xs):
        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()

        if np.shape(xs)[0] == 1:
            ax = axes[0][0]
        else:
            if reverse:
                ax = axes[K-i-1, K-i-1]
            else:
                ax = axes[0, i]
        if show_titles:
            title = None
            if title_fmt is not None:
                # Compute the quantiles for the title. This might redo
                # unneeded computation but who cares.

                if use_hpd == True:
                    my_mode = mode(x)
                    my_hpd = hpd(x,conf=0.68)
                    q_50 = my_mode
                    q_16, q_84 = my_hpd
                    d = my_hpd - my_mode
                    dd = np.min(abs(d))
                    ld = np.log10(dd)
                    lld = int(-1*np.sign(ld)*(math.ceil(abs(ld))) )
                    if lld <0:
                        lld += 1

                    lld += (sig_fig-1)

                    rm = np.round(my_mode,lld)
                    re = np.round(d,lld)
                    title = "${:.{prec}f}_{{{:.{prec}f}}}^{{+{:.{prec}f}}}$".format(rm,re[0],re[1],prec=lld)
                else:
                    q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84],
                                                weights=weights)
                    q_m, q_p = q_50-q_16, q_84-q_50
                    # Format the quantile display.
                    fmt = "{{0:{0}}}".format(title_fmt).format
                    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                    title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))

                # Add in the column name if it's given.
                if labels is not None:
                    if units is not None:
#                        title = "{0} = {1} {2}".format(labels[i], title,units[i])
                        title = "{1} {2}".format(labels[i], title,units[i])
                    else:
#                        title = "{0} = {1}".format(labels[i], title)
                        title = "{1}".format(labels[i], title)

            elif labels is not None:
                title = "{0}".format(labels[i])

            if title is not None:
                if reverse:
                    ax.set_xlabel(title, **title_kwargs)
                else:
                    ax.set_title(title, **title_kwargs)


    for i, x in enumerate(xs):
        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()

        if np.shape(xs)[0] == 1:
            ax = axes[0][0]
        else:
            if reverse:
                ax = axes[K-i-1, K-i-1]
            else:
                ax = axes[i, i]
        # Plot the histograms.
        if smooth1d is None:
            n, _, _ = ax.hist(x, bins=bins[i], weights=weights,
                              range=np.sort(drange[i]), **hist_kwargs)
        else:
            if gaussian_filter is None:
                raise ImportError("Please install scipy for smoothing")
            n, b = np.histogram(x, bins=bins[i], weights=weights,
                                range=np.sort(drange[i]))
            n = gaussian_filter(n, smooth1d)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
            ax.plot(x0, y0, **hist_kwargs)

        if truths is not None and truths[i] is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            if use_hpd == True:
                lss = ["dashed","dotted"]
                lsi = 0
                for q in quantiles:
                    myqs = hpd(x,conf=q)
                    mymode = mode(x)
                    for qq in myqs:
                        ax.axvline(qq, ls=lss[lsi], color=color)
                    ax.axvline(mymode, ls="solid", color=color)
                    lsi += 1
            else:
                qvalues = quantile(x, quantiles, weights=weights)
                for q in qvalues:
                    ax.axvline(q, ls="dashed", color=color)

            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

        ax.plot(priors[0][i],priors[1][i],color=get_cmap('viridis')(0),zorder=-1)
        ax.plot(priors[0][i],priors[1][i],color=get_cmap('viridis')(0),zorder=3,alpha=0.5)
        ax.fill(priors[0][i],priors[1][i],color=(0.26700400000000002*0.5 +0.5, 0.0048739999999999999*0.5 +0.5, 0.32941500000000001*0.5 +0.5, 1.0),zorder=-1)

        # Set up the axes.

        if set_lims == "prior":
            ax.set_xlim(prange[i])
        else:
            ax.set_xlim(drange[i])

        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.1 * maxn)
        else:
            ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(NullLocator())

        if i < K - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            if reverse:
                ax.xaxis.tick_top()
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                if units is not None:
                    if units[i] == "":
                        label = "{}".format(labels[i])
                    else:
                        label = "{} [{}]".format(labels[i],units[i])
                else:
                    label = "{}".format(labels[i])

                if reverse:
                    ax.set_title(label, y=1.25, **label_kwargs)
                else:
                    ax.set_xlabel(label, **label_kwargs)

            # use MathText for axes ticks
            ax.xaxis.set_major_formatter(
                ScalarFormatter(useMathText=use_math_text))

        for j, y in enumerate(xs):
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                if reverse:
                    ax = axes[K-i-1, K-j-1]
                    ax2 = axes[K-j-1, K-i-1]
                else:
                    ax = axes[i, j]
                    ax2 = axes[j, i]
            if priors == []:
                if j > i:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
            if j == i:
                continue

            # Deal with masked arrays.
            if hasattr(y, "compressed"):
                y = y.compressed()

            else:
                if j < i:

                    p1 = priors[1][j][1:-1]
                    p2 = priors[1][i][1:-1]

                    md1 = np.min(np.diff(priors[0][j]))
                    rr1 = np.arange(np.min(priors[0][j][np.isfinite(priors[0][j])]),np.max(priors[0][j][np.isfinite(priors[0][j])]),md1)
                    f1 = interpolate.interp1d(priors[0][j][np.isfinite(priors[0][j])], priors[1][j][np.isfinite(priors[0][j])],kind="cubic")
                    new1 = f1(rr1)

                    md2 = np.min(np.diff(priors[0][i]))
                    rr2 = np.arange(np.min(priors[0][i][np.isfinite(priors[0][i])]),np.max(priors[0][i][np.isfinite(priors[0][i])]),md2)
                    f2 = interpolate.interp1d(priors[0][i][np.isfinite(priors[0][i])], priors[1][i][np.isfinite(priors[0][i])],kind="cubic")
                    new2 = f2(rr2)

                    X2, Y2, H2, V, r1, r2 = hist2d(y, x, ax=ax, drange=[drange[j], drange[i]], weights=weights,
                           color=color, smooth=smooth, bins=[bins[j], bins[i]], hex=hex, prange=[prange[j],prange[i]],
                           **hist2d_kwargs)

                    bl = np.outer(new2,new1)

                    p1f = False
                    p2f = False
                    if ((round(p1[1],2) == round(p1[-2],2)) & (round(p1[1],2) == round(p1[int(len(p1)/2)],2) )):
                        p1f = True
                    if ((round(p2[1],2) == round(p2[-2],2)) & (round(p2[1],2) == round(p2[int(len(p2)/2)],2) )):
                        p2f = True

                    density_cmap = viridis_r
                    my_cmap = density_cmap(np.arange(density_cmap.N))
                    my_cmap[:,-1] = np.linspace(1,0,density_cmap.N)
                    density_cmap = ListedColormap(my_cmap)

                    xedge = np.hstack((rr2[:-1]-0.5*np.diff(rr2), rr2[-2:] +0.5*np.diff(rr2)[-1]))
                    yedge = np.hstack((rr1[:-1]-0.5*np.diff(rr1), rr1[-2:] +0.5*np.diff(rr1)[-1]))

                    HP2, XP2, YP2, VP = gen_contours(bl,yedge,xedge)

                    ax2.set_facecolor(density_cmap(0.9))
                    if (p1f == False) & (p2f == False):
                        ax2.contourf(YP2, XP2, HP2.T, [VP[0],VP[1]],colors=["white"],antialiased=False)
                        ax2.contourf(YP2, XP2, HP2.T, [VP[0],VP[1]],colors=[density_cmap(0.5)],antialiased=False)
                        ax2.contourf(YP2, XP2, HP2.T, [VP[1],HP2.max()*(1+1e-4)],colors=[density_cmap(0.0)],antialiased=False)
                    if (p1f == True) & (p2f == True):
                        ax2.set_facecolor(density_cmap(0.0))
                    if (p1f == True) & (p2f == False):
                        sm = np.cumsum(new2)
                        sm /= sm[-1]
                        in1 = np.argmin(abs(sm-0.025))
                        in2 = np.argmin(abs(sm-0.975))
                        in3 = np.argmin(abs(sm-0.16))
                        in4 = np.argmin(abs(sm-0.84))
                        ax2.axvspan(rr2[in1],rr2[in2],color=density_cmap(0.5))
                        ax2.axvspan(rr2[in3],rr2[in4],color=density_cmap(0.0))

                    if (p1f == False) & (p2f == True):
                        sm = np.cumsum(new1)
                        sm /= sm[-1]
                        in1 = np.argmin(abs(sm-0.025))
                        in2 = np.argmin(abs(sm-0.975))
                        in3 = np.argmin(abs(sm-0.16))
                        in4 = np.argmin(abs(sm-0.84))
                        ax2.axhspan(rr1[in1],rr1[in2],color=density_cmap(0.5))
                        ax2.axhspan(rr1[in3],rr1[in4],color=density_cmap(0.0))

                    # Plot the contour edge colors.
                    plot_contours = True
                    contour_kwargs = None
                    if plot_contours:
                        if contour_kwargs is None:
                            contour_kwargs = dict()
                        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
                        contour_kwargs["colors"] = [viridis_r(1.0),viridis_r(1.0)]
                        ax2.contour(Y2, X2, H2, V, **contour_kwargs)

                    if set_lims == "prior":
                        ax2.set_xlim(prange[i])
                        ax2.set_ylim(prange[j])
                        ax.set_xlim(prange[j])
                        ax.set_ylim(prange[i])

                    else:
                        ax2.set_xlim(r2)
                        ax2.set_ylim(r1)
                        ax.set_xlim(r1)
                        ax.set_ylim(r2)

            if truths is not None:
                if truths[i] is not None and truths[j] is not None:
                    ax.plot(truths[j], truths[i], "s", color=truth_color)
                if truths[j] is not None:
                    ax.axvline(truths[j], color=truth_color)
                if truths[i] is not None:
                    ax.axhline(truths[i], color=truth_color)

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                if reverse:
                    ax.xaxis.tick_top()
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    if units is not None:
                        if units[j] == "":
                            label = "{}".format(labels[j])
                        else:
                            label = "{} [{}]".format(labels[j],units[j])
                    else:
                        label = "{}".format(labels[j])

                    ax.set_xlabel(label, **label_kwargs)
                    if reverse:
                        ax.xaxis.set_label_coords(0.5, 1.4)
                    else:
                        ax.xaxis.set_label_coords(0.5, -0.3)

                # use MathText for axes ticks
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text))

            if j > 0:
                ax.set_yticklabels([])
            else:
                if reverse:
                    ax.yaxis.tick_right()
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    if units is not None:
                        if units[i] == "":
                            label = "{}".format(labels[i])
                        else:
                            label = "{} [{}]".format(labels[i],units[i])
                    else:
                        label = "{}".format(labels[i])

                    if reverse:
                        ax.set_ylabel(label, rotation=-90, **label_kwargs)
                        ax.yaxis.set_label_coords(1.3, 0.5)
                    else:
                        ax.set_ylabel(label, **label_kwargs)
                        ax.yaxis.set_label_coords(-0.3, 0.5)

                # use MathText for axes ticks
                ax.yaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text))

    return fig


def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the drange
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()

def gen_contours(H,X,Y,smooth=None,levels=None):

    # Choose the default "sigma" contour levels.
    if levels is None:
#        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
        levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2)

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        logging.warning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
    ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
    ])

    return H2, X2, Y2, V

def hist2d(x, y, bins=20, drange=None, weights=None, levels=None, smooth=None,
           ax=None, color=None, plot_datapoints=True, plot_density=True,
           plot_contours=True, no_fill_contours=False, fill_contours=False,
           hex=True, contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
           **kwargs):
    """
    Plot a 2-D histogram of samples.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    y : array_like[nsamples,]
       The samples.

    levels : array_like
        The contour levels to draw.

    ax : matplotlib.Axes
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool
        Draw the individual data points.

    plot_density : bool
        Draw the density colormap.

    plot_contours : bool
        Draw the contours.

    no_fill_contours : bool
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool
        Fill the contours.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    """

    if ax is None:
        ax = pl.gca()

    # Set the default drange based on the data drange if not provided.
    if drange is None:
        if "extent" in kwargs:
            logging.warn("Deprecated keyword argument 'extent'. "
                         "Use 'range' instead.")
            drange = kwargs["extent"]
        else:
            drange = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "C1"

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])
    density_cmap = viridis_r
    my_cmap = density_cmap(np.arange(density_cmap.N))
    my_cmap[:,-1] = np.linspace(1,0,density_cmap.N)
    density_cmap = ListedColormap(my_cmap)

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.

    # Choose the default "sigma" contour levels.
    if levels is None:
#        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
        levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2)


    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, drange)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic drange. You could try using the "
                         "'range' argument.")

    H2, X2, Y2, V = gen_contours(H,X,Y,smooth=smooth,levels=levels)

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["color"] = viridis_r(1.0)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        if hex == True:
            hexes = ax.hexbin(x.flatten(), y.flatten(), cmap=density_cmap.reversed(),gridsize=[int(b/2) for b in bins])
        else:
            ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)


    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        contour_kwargs["colors"] = [viridis_r(1.0),viridis_r(1.0)]
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    return X2, Y2, H2, V, drange[0], drange[1]
