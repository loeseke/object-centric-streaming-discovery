"""\
Functionality for evaluating data collected by RuntimeMonitor or CacheMonitor during stream processing.
__author__: "Nina Löseke"
"""

from typing import Union
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
from pathlib import Path
from utils import *
from monitor import CacheMonitor, RuntimeMonitor
from model_buffers import TotemBuffer, OcdfgBuffer, OcdfgBufferPerObjectType, OcpnBuffer
from cache_policy_buffers import CachePolicy, BufferOfDicts, BufferOfDictLists
from priority_policy_buffers import PPBCustom, PPBEventsPerObjectType, PPBLifespanPerObject, PPBLifespanPerObjectType, PPBObjectsPerEvent, PPBObjectsPerObjectType, PPBStridePerObject, PPBStridePerObjectType, PrioPolicyBuffer, PrioPolicyOrder
from vars import *


def plot_buf_update_rt(rt_mon : RuntimeMonitor, output_dir : Path) -> None:
    """
    Plots lineplot of moving average of buffer-update times per model buffer over course of stream and
    plots boxplot of average buffer-update times per model buffer.

    Parameters
    ----------
    rt_mon : RuntimeMonitor
        RuntimeMonitor object for which DataFrame has already been created for plotting.
    output_dir : Path
        Output directory to which PDF is saved.

    Returns
    -------
    None
    """
    rt_buf_df = rt_mon.buf_update_times_df
    window_size = 5000

    bufs = sorted(rt_buf_df[M_BUF_NAME].unique())
    buf_colors = cm.cool(np.linspace(0, 1, len(bufs)))
    buf_to_color = dict(zip(bufs, buf_colors))

    # Create lineplot of buffer-update times over stream items
    fig_0, ax_0 = plt.subplots(1, 1, figsize=(5, 4))
    fig_0.tight_layout()
    ax_0.set_title(f'Moving average of buffer-update runtime (window size {window_size})')
    ax_0.set_xlabel('Stream item')
    ax_0.set_ylabel('Avg. update time [ms]')

    for buf in bufs:
        rt_buf_df_buf = rt_buf_df.loc[rt_buf_df[M_BUF_NAME] == buf].drop(M_BUF_NAME, axis=1)
        rt_buf_df_buf['moving average'] = rt_buf_df_buf[M_BUF_UPDATE_TIME].rolling(window=window_size).mean()
        ax_0.plot(rt_buf_df_buf[M_STREAM_ITEM], rt_buf_df_buf['moving average'], color=buf_to_color[buf], label=buf)
    ax_0.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.36))

    os.makedirs(output_dir, exist_ok=True)
    fig_0.savefig(os.path.join(output_dir, 'runtime_per_buf_lineplot.pdf'), format='pdf', bbox_inches='tight')

    # Create boxplot of buffer-update times
    fig_1, ax_1 = plt.subplots(1, 1, figsize=(5, 4))
    fig_1.tight_layout()
    ax_1.set_title(f'Boxplot of buffer-update runtime')
    ax_1.set_xlabel('Model buffer')
    ax_1.set_ylabel('Update time [ms]')

    sns.boxplot(data=rt_buf_df, x=M_BUF_NAME, y=M_BUF_UPDATE_TIME, ax=ax_1, order=bufs, hue=M_BUF_NAME, palette=buf_to_color, legend=None)
    ax_1.tick_params(axis='x', labelrotation=40)

    fig_1.savefig(os.path.join(output_dir, 'runtime_per_buf_boxplot.pdf'), format='pdf', bbox_inches='tight')


def plot_stream_item_processing_rt(rt_mon : RuntimeMonitor, output_dir : Path) -> None:
    """
    Plots lineplot of moving average of stream-item processing time per item type and
    plots pie chart of average stream-item processing time per item type.

    Parameters
    ----------
    rt_mon : RuntimeMonitor
        RuntimeMonitor object for which DataFrame has already been created for plotting.
    output_dir : Path
        Output directory to which PDF is saved.

    Returns
    -------
    None
    """
    rt_item_df = rt_mon.item_processing_times_df
    rt_buf_df = rt_mon.buf_update_times_df

    bufs = sorted(rt_buf_df[M_BUF_NAME].unique())
    buf_colors = cm.cool(np.linspace(0, 1, len(bufs)))
    buf_to_color = dict(zip(bufs, buf_colors))

    item_types = sorted(['Event', 'O2OUpdate', 'ObjectAttributeUpdate']) # sorted(rt_item_df[M_ITEM_TYPE].unique())
    item_colors = cm.autumn(np.linspace(0, 1, len(item_types)))
    item_to_color = dict(zip(item_types, item_colors))

    # Create lineplot of moving average of stream-item processing time per item type (Event, O2OUpdate etc.)
    window_size = 4000
    fig_0, ax_0 = plt.subplots(1, 1, figsize=(5, 4))
    fig_0.tight_layout()
    ax_0.set_title(f'Moving average of stream-item processing time (window size {window_size})')
    ax_0.set_xlabel('Stream item')
    ax_0.set_ylabel('Avg. processing time [ms]')

    for item in item_types:
        rt_item_df_item = rt_item_df.loc[rt_item_df[M_ITEM_TYPE] == item].drop(M_ITEM_TYPE, axis=1)
        rt_item_df_item['moving average'] = rt_item_df_item[M_ITEM_PROCESSING_TIME].rolling(window=window_size).mean()
        ax_0.plot(rt_item_df_item[M_STREAM_ITEM], rt_item_df_item['moving average'], color=item_to_color[item], label=item)
    ax_0.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.3))

    os.makedirs(output_dir, exist_ok=True)
    fig_0.savefig(os.path.join(output_dir, 'runtime_per_stream_item_lineplot.pdf'), format='pdf', bbox_inches='tight')

    # Create piechart of % of buffer-update times and buffer-item-creation time for avg. processing of a stream item
    fig_1, ax_1 = plt.subplots(1, 1, figsize=(5, 4))
    fig_1.tight_layout()
    ax_1.set_title(f'Split of avg. per-stream-item processing time')

    num_stream_items = rt_item_df[M_STREAM_ITEM].nunique()
    avg_item_rt = rt_item_df[M_ITEM_PROCESSING_TIME].sum() / num_stream_items
    avg_item_creation_rt = avg_item_rt
    avg_buf_update_times = list()
    for buf in bufs:
        rt_buf_df_buf = rt_buf_df.loc[rt_buf_df[M_BUF_NAME] == buf]
        avg_buf_update_rt = rt_buf_df_buf[M_BUF_UPDATE_TIME].sum() / num_stream_items
        avg_item_creation_rt -= avg_buf_update_rt
        avg_buf_update_times.append(avg_buf_update_rt)
    pie_labels = ['buffer-item creation'] + bufs
    pie_values = [avg_item_creation_rt] + avg_buf_update_times
    pie_colors = ['lightgray'] + [buf_to_color[buf] for buf in bufs]
    ax_1.pie(pie_values, labels=None, colors=pie_colors, autopct='%1.0f%%')

    ax_1.legend(title=f'Avg. stream-item processing time: {avg_item_rt:.2f} ms', labels=pie_labels, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.3))

    fig_1.savefig(os.path.join(output_dir, 'runtime_per_stream_item_pie.pdf'), format='pdf', bbox_inches='tight')


def plot_cache_stats_per_ot_over_stream(c_mon : CacheMonitor, output_dir : Path) -> None:
    """
    Plots lineplot of cumulative cache hits/misses/evictions per object type over course of stream.

    Parameters
    ----------
    c_mon : CacheMonitor
        CacheMonitor object for which DataFrame has already been created for plotting.
    output_dir : Path
        Output directory to which PDF is saved.
    
    Returns
    -------
    None
    """
    # Cache hits/misses/evictions per object type across all model buffers
    hits_df = c_mon.hits_df
    evic_df = c_mon.evic_df.set_index(M_STREAM_ITEM)
    evic_df['cum_count'] = evic_df.groupby(M_OBJECT_TYPE).cumcount() # + 1

    ots = sorted(hits_df[M_OBJECT_TYPE].unique())
    ot_colors = cm.jet(np.linspace(0, 1, len(ots)))
    ot_to_color = dict(zip(ots, ot_colors))

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)

    axs[1].set_title('Cache hits per object type across all model buffers')
    axs[1].set_xlabel('Stream item')
    axs[1].set_ylabel('Cumulative cache hits')
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_tick_params(labelright=True)

    axs[0].set_title('Cache evictions per object type across all model buffers')
    axs[0].set_xlabel('Stream item')
    axs[0].set_ylabel('Cumulative cache evictions')
    axs[0].yaxis.tick_right()

    axs[2].set_title('Cache misses per object type across all model buffers')
    axs[2].set_xlabel('Stream item')
    axs[2].set_ylabel('Cumulative cache misses')
    axs[2].yaxis.tick_right()
    axs[2].yaxis.set_tick_params(labelright=True)

    for ot in ots:
        hits_df_ot = hits_df.loc[hits_df[M_OBJECT_TYPE] == ot]
        evic_df_ot = evic_df.loc[evic_df[M_OBJECT_TYPE] == ot]

        x_hits_misses = hits_df_ot[M_STREAM_ITEM]
        y_hits = hits_df_ot[M_CACHE_HITS].cumsum()
        axs[0].plot(x_hits_misses, y_hits, color=ot_to_color[ot], label=ot)

        x_evic = evic_df_ot.index
        y_evic = evic_df_ot['cum_count']
        axs[1].plot(x_evic, y_evic, color=ot_to_color[ot], label=ot)
        
        y_misses = hits_df_ot[M_CACHE_MISSES].cumsum()
        axs[2].plot(x_hits_misses, y_misses, color=ot_to_color[ot], label=ot)

    lgd_handles = [Line2D([0], [0], color=ot_to_color[ot], label=ot) for ot in ot_to_color]
    fig.legend(loc='lower center', handles=lgd_handles, ncol=len(ots), bbox_to_anchor=(0.5, -0.15))

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'cache_stats_per_ot_lineplot.pdf'), format='pdf', bbox_inches='tight')


def plot_cache_stats_per_ot_total(c_mon : CacheMonitor, output_dir : Path) -> None:
    """
    Plots bar plot of total cache hits/misses/evictions per object type across all model buffers.

    Parameters
    ----------
    c_mon : CacheMonitor
        CacheMonitor object for which DataFrame has already been created for plotting.
    output_dir : Path
        Output directory to which PDF is saved.
    
    Returns
    -------
    None
    """
    # Grouped bar plot for cache hits/evictions/misses (sum) per object type across all buffers
    hits_df = c_mon.hits_df
    evic_df = c_mon.evic_df.set_index(M_STREAM_ITEM)
    evic_df['cum_count'] = evic_df.groupby(M_OBJECT_TYPE).cumcount() # + 1

    ot_to_hits_misses = hits_df.groupby(M_OBJECT_TYPE)[[M_CACHE_HITS, M_CACHE_MISSES]].sum()
    ot_to_evic = evic_df.groupby(M_OBJECT_TYPE)[[M_BUF_NAME]].count() #+1
    ot_to_evic = ot_to_evic.rename(columns={M_BUF_NAME: '# evictions'})
    ot_to_hme = pd.merge(ot_to_evic, ot_to_hits_misses, left_index=True, right_index=True).fillna(0)

    ot_to_hme = ot_to_hme.reset_index()
    ot_to_hme = ot_to_hme.melt(id_vars=[M_OBJECT_TYPE], var_name='type', value_name='sum')

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.tight_layout()

    ax.set_title('Cache statistics per object type across all model buffers')
    ax.set_xlabel('Object type')
    ax.set_ylabel('Total')
    ax.tick_params(axis='x', labelrotation=40)

    ax = sns.barplot(x=M_OBJECT_TYPE, y='sum', hue='type', data=ot_to_hme, palette=sns.color_palette('coolwarm', 3))
    ax.get_legend().set_title(None)
    sns.move_legend(ax, loc="upper right")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'cache_stats_per_ot_barplot.pdf'), format='pdf', bbox_inches='tight')


def plot_cache_stats_per_buf_over_stream(c_mon : CacheMonitor, output_dir : Path) -> None:
    """
    Plots lineplot of cumulative cache hits/misses/evictions per model buffer over course of stream.

    Parameters
    ----------
    c_mon : CacheMonitor
        CacheMonitor object for which DataFrame has already been created for plotting.
    output_dir : Path
        Output directory to which PDF is saved.
    
    Returns
    -------
    None
    """
    # Cache hits/misses/evictions per model buffer across all object types
    hits_df = c_mon.hits_df
    evic_df = c_mon.evic_df.set_index(M_STREAM_ITEM)
    evic_df['cum_count'] = evic_df.groupby(M_BUF_NAME).cumcount() # + 1

    bufs = sorted(hits_df[M_BUF_NAME].unique())
    buf_colors = cm.cool(np.linspace(0, 1, len(bufs)))
    buf_to_color = dict(zip(bufs, buf_colors))

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)

    axs[1].set_title('Cache hits per model buffer across all object types')
    axs[1].set_xlabel('Stream item')
    axs[1].set_ylabel('Cumulative cache hits')
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_tick_params(labelright=True)

    axs[0].set_title('Cache evictions per model buffer across all object types')
    axs[0].set_xlabel('Stream item')
    axs[0].set_ylabel('Cumulative cache evictions')
    axs[0].yaxis.tick_right()

    axs[2].set_title('Cache misses per model buffer across all object types')
    axs[2].set_xlabel('Stream item')
    axs[2].set_ylabel('Cumulative cache misses')
    axs[2].yaxis.tick_right()
    axs[2].yaxis.set_tick_params(labelright=True)

    for buf in bufs:
        hits_df_buf = hits_df.loc[hits_df[M_BUF_NAME] == buf]
        evic_df_buf = evic_df.loc[evic_df[M_BUF_NAME] == buf]

        x_hits_misses = hits_df_buf[M_STREAM_ITEM]
        y_hits = hits_df_buf[M_CACHE_HITS].cumsum()
        axs[1].plot(x_hits_misses, y_hits, color=buf_to_color[buf], label=buf)

        x_evic = evic_df_buf.index
        y_evic = evic_df_buf['cum_count']
        axs[0].plot(x_evic, y_evic, color=buf_to_color[buf], label=buf)

        y_misses = hits_df_buf[M_CACHE_MISSES].cumsum()
        axs[2].plot(x_hits_misses, y_misses, color=buf_to_color[buf], label=buf)

    lgd_handles = [Line2D([0], [0], color=buf_to_color[buf], label=buf) for buf in buf_to_color]
    fig.legend(loc='lower center', handles=lgd_handles, ncol=len(bufs), bbox_to_anchor=(0.5, -0.15))

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'cache_stats_per_buf_lineplot.pdf'), format='pdf', bbox_inches='tight')


def plot_cache_stats_per_buf_total(c_mon : CacheMonitor, output_dir : Path) -> None:
    """
    Plots bar plot of total cache hits/misses/evictions per model buffer across all object types.

    Parameters
    ----------
    c_mon : CacheMonitor
        CacheMonitor object for which DataFrame has already been created for plotting.
    output_dir : Path
        Output directory to which PDF is saved.
    
    Returns
    -------
    None
    """
    # Grouped bar plot for cache hits/evictions/misses (sum) per bufffer across all object types
    hits_df = c_mon.hits_df
    evic_df = c_mon.evic_df.set_index(M_STREAM_ITEM)
    evic_df['cum_count'] = evic_df.groupby(M_BUF_NAME).cumcount() # + 1

    ots = sorted(hits_df[M_OBJECT_TYPE].unique())
    ot_colors = cm.jet(np.linspace(0, 1, len(ots)))
    ot_to_color = dict(zip(ots, ot_colors))

    ot_to_hits_misses = hits_df.groupby(M_BUF_NAME)[[M_CACHE_HITS, M_CACHE_MISSES]].sum()
    ot_to_evic = evic_df.groupby(M_BUF_NAME)[[M_OBJECT_TYPE]].count() #+1
    ot_to_evic = ot_to_evic.rename(columns={M_OBJECT_TYPE: '# evictions'})
    ot_to_hme = pd.merge(ot_to_evic, ot_to_hits_misses, left_index=True, right_index=True).fillna(0)

    ot_to_hme = ot_to_hme.reset_index()
    ot_to_hme = ot_to_hme.melt(id_vars=[M_BUF_NAME], var_name='type', value_name='sum')
    ot_to_hme['color'] = ot_to_hme[M_BUF_NAME].map(ot_to_color)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.tight_layout()

    ax.set_title('Cache statistics per model buffer across all object types')
    ax.set_xlabel('Model buffer')
    ax.set_ylabel('Total')
    ax.tick_params(axis='x', labelrotation=0)

    cache_stats_cp = sns.color_palette("coolwarm", 3)

    ax = sns.barplot(x=M_BUF_NAME, y='sum', data=ot_to_hme, palette=cache_stats_cp, hue='type')
    ax.get_legend().set_title(None)
    sns.move_legend(ax, loc="upper right")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'cache_stats_per_buf_barplot.pdf'), format='pdf', bbox_inches='tight')


def plot_hit_miss_prob(c_mon : CacheMonitor, output_dir : Path) -> None:
    """
    Plots stacked bar plot of cache hit vs. miss probability per model buffer and per object type.

    Parameters
    ----------
    c_mon : CacheMonitor
        CacheMonitor object for which DataFrame has already been created for plotting.
    output_dir : Path
        Output directory to which PDF is saved.

    Returns
    -------
    None
    """
    # % hits-to-misses per stream item
    hits_df = c_mon.hits_df
    buf_groups = hits_df.groupby(M_BUF_NAME)
    buf_to_df = dict([(name, df) for name, df in buf_groups])
    ot_groups = hits_df.groupby(M_OBJECT_TYPE)
    ot_to_df = dict([(name, df) for name, df in ot_groups])

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=False, sharey=True)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)

    plot_buf_df_rows = list()
    plot_ot_df_rows = list()

    for buf, buf_df in buf_to_df.items():
        buf_df = buf_df.drop([M_BUF_NAME, M_OBJECT_TYPE], axis=1)
        buf_df = buf_df.groupby(M_STREAM_ITEM).sum()
        buf_df = buf_df.apply(lambda row: row / (row[M_CACHE_MISSES] + row[M_CACHE_HITS]), axis=1)
        mean_series = buf_df.mean(axis=0)
        plot_buf_df_rows.append([buf, mean_series[M_CACHE_HITS], mean_series[M_CACHE_MISSES]])

    for ot, ot_df in ot_to_df.items():
        ot_df = ot_df.drop([M_BUF_NAME, M_OBJECT_TYPE], axis=1)
        ot_df = ot_df.groupby(M_STREAM_ITEM).sum()
        ot_df = ot_df.apply(lambda row: row / (row[M_CACHE_MISSES] + row[M_CACHE_HITS]), axis=1)
        mean_series = ot_df.mean(axis=0)
        plot_ot_df_rows.append([ot, mean_series[M_CACHE_HITS], mean_series[M_CACHE_MISSES]])

    barplot_buf_df = pd.DataFrame(columns=[M_BUF_NAME, M_CACHE_HITS, M_CACHE_MISSES], data=plot_buf_df_rows).set_index(M_BUF_NAME)
    barplot_ot_df = pd.DataFrame(columns=[M_OBJECT_TYPE, M_CACHE_HITS, M_CACHE_MISSES], data=plot_ot_df_rows).set_index(M_OBJECT_TYPE)

    # Make sure colors are the same as in seaborn plot

    barplot_buf_df.plot(kind='bar', stacked=True, ax=axs[0], color=sns.color_palette("coolwarm", 3).as_hex()[1:], legend=None)
    barplot_ot_df.plot(kind='bar', stacked=True, ax=axs[1], color=sns.color_palette("coolwarm", 3).as_hex()[1:], legend=None)

    axs[0].tick_params(axis='x', labelrotation=40)
    axs[0].set_title(f'Avg. hit-miss probability per stream item per model buffer')
    axs[0].set_ylabel('Fraction')
    axs[0].set_xlabel('Model buffer')

    axs[1].tick_params(axis='x', labelrotation=40)
    axs[1].set_title(f'Avg. hit-miss probability per stream item per object type')
    axs[1].set_ylabel('Fraction')
    axs[1].set_xlabel('Object type')
    axs[1].yaxis.set_tick_params(labelleft=True)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(loc="lower center", handles=handles, labels=labels, bbox_to_anchor=(0.5, -0.33), ncol=2)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'cache_hit_miss_prob_barplot.pdf'), format='pdf', bbox_inches='tight')


def plot_full_cache_evic_over_stream(c_mon : CacheMonitor, output_dir : Path) -> None:
    """
    Plots lineplot of cumulative total cache evictions (i.e. no corresponding items left in model buffer) per object type and per model buffer.

    Parameters
    ----------
    c_mon : CacheMonitor
        CacheMonitor object for which DataFrame has already been created for plotting.
    output_dir : Path
        Output directory to which PDF is saved.
    
    Returns
    -------
    None
    """
    # Corresponding DataFrame may be empty, e.g. for OcdfgBufferPerObjectType
    if len(c_mon.full_evic_df) == 0:
        return
    
    # Full evictions per object type and per model buffer
    full_evic_df = c_mon.full_evic_df.set_index(M_STREAM_ITEM)
    full_evic_df['cum_count_ot'] = full_evic_df.groupby(M_OBJECT_TYPE).cumcount() # + 1
    full_evic_df['cum_count_buf'] = full_evic_df.groupby(M_BUF_NAME).cumcount() # + 1

    ots = sorted(full_evic_df[M_OBJECT_TYPE].unique())
    ot_colors = cm.jet(np.linspace(0, 1, len(ots)))
    ot_to_color = dict(zip(ots, ot_colors))

    bufs = sorted(full_evic_df[M_BUF_NAME].unique())
    buf_colors = cm.cool(np.linspace(0, 1, len(bufs)))
    buf_to_color = dict(zip(bufs, buf_colors))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)

    axs[0].set_title('Full cache evictions per object type across all model buffers')
    axs[0].set_xlabel('Stream item')
    axs[0].set_ylabel('Cumulative full cache evictions')
    axs[0].yaxis.tick_right()

    axs[1].set_title('Full cache evictions per model buffer across all object types')
    axs[1].set_xlabel('Stream item')
    axs[1].set_ylabel('Cumulative full cache evictions')
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_tick_params(labelright=True)

    for ot in ots:
        full_evic_df_ot = full_evic_df.loc[full_evic_df[M_OBJECT_TYPE] == ot]
        x_ot = full_evic_df_ot.index
        y_ot = full_evic_df_ot['cum_count_ot']
        axs[0].plot(x_ot, y_ot, color=ot_to_color[ot], label=ot)

    for buf in bufs:
        full_evic_df_buf = full_evic_df.loc[full_evic_df[M_BUF_NAME] == buf]
        x_buf = full_evic_df_buf.index
        y_buf = full_evic_df_buf['cum_count_buf']
        axs[1].plot(x_buf, y_buf, color=buf_to_color[buf], label=buf)

    axs[0].legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.4))
    axs[1].legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.3))

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'full_cache_evictions_lineplot.pdf'), format='pdf', bbox_inches='tight')


def plot_ot_frac_per_buf(c_mon : CacheMonitor, output_dir : Path) -> None:
    """
    Plots stacked bar plot per model buffer of fraction of items per object type over course of stream.

    Parameters
    ----------
    c_mon : CacheMonitor
        CacheMonitor object for which DataFrame has already been created for plotting.
    output_dir : Path
        Output directory to which PDF is saved.

    Returns
    -------
    None
    """
    # % OT per model buffer over time
    ot_frac_df = c_mon.ot_frac_df
    ot_frac_df['pct'] = ot_frac_df[M_STREAM_ITEM].div(ot_frac_df[M_STREAM_ITEM].max()).mul(100).round(0)
    ot_frac_df = ot_frac_df.set_index('pct')
    ot_frac_df = ot_frac_df.sort_values(by=[M_OBJECT_TYPE])

    ot_frac_df = ot_frac_df[[M_OT_PERCENTAGE, M_OBJECT_TYPE, M_BUF_NAME]]

    bufs = sorted(ot_frac_df[M_BUF_NAME].unique())
    ots = sorted(ot_frac_df[M_OBJECT_TYPE].unique())
    # NOTE: fixed to ContainerLogistics log
    ot_to_color_full = dict(zip(['Container', 'Customer Order', 'Forklift', 'Handling Unit', 'Transport Document', 'Truck', 'Vehicle'], cm.jet(np.linspace(0, 1, 7))))
    ot_to_color = {ot: ot_to_color_full[ot] for ot in ots}

    fig, axs = plt.subplots(len(bufs), 1, figsize=(5, 3*len(bufs)), sharex=True, sharey=True)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)

    buf_groups = ot_frac_df.groupby(M_BUF_NAME)
    buf_to_df = dict([(name, df) for name, df in buf_groups])

    for i_buf, (buf, buf_df) in enumerate(buf_to_df.items()):
        buf_df = buf_df.drop(M_BUF_NAME, axis=1)
        buf_df = buf_df.sort_values(by=[M_OBJECT_TYPE])
        buf_df = buf_df.pivot(columns=M_OBJECT_TYPE, values=M_OT_PERCENTAGE).fillna(0)
        buf_df.reset_index()

        buf_df.plot(kind='bar', stacked=True, ax=axs[i_buf], label=M_OBJECT_TYPE, color=ot_to_color, legend=None)
        axs[i_buf].tick_params(axis='x', labelrotation=0)
        axs[i_buf].set_title(f'Fractions of object types in {buf}')
        axs[i_buf].set_ylabel('Fraction per object type')
        axs[i_buf].set_xlabel('Position in stream [%]')
        axs[i_buf].yaxis.set_tick_params(labelleft=True)

    lgd_handles = [Patch(facecolor=ot_to_color[ot], label=ot) for ot in ot_to_color]
    fig.legend(loc="lower center", handles=lgd_handles, bbox_to_anchor=(0.5, -0.12), ncol=2)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'ot_fractions_per_buf_barplot.pdf'), format='pdf', bbox_inches='tight')


def plot_pp_buf_size_over_stream(c_mon : CacheMonitor, output_dir : Path) -> None:
    """
    Plots moving average of size of priority-policy buffer over course of stream.

    Parameters
    ----------
    c_mon : CacheMonitor
        CacheMonitor object for which DataFrame has already been created for plotting.
    output_dir : Path
        Output directory to which PDF is saved.

    Returns
    -------
    None
    """
    # Corresponding DataFrame might be empty, e.g. if no PPB is None
    if len(c_mon.ppb_size_df) == 0:
        return
    
    # Priority-policy buffer size over time
    ppb_size_df = c_mon.ppb_size_df
    window_size = 2000
    ppb_size_df['moving average'] = ppb_size_df[M_PPB_SIZE].rolling(window=window_size).mean()
    ppb_size_df.dropna()

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.tight_layout()

    x = ppb_size_df.index
    y = ppb_size_df['moving average']

    fig.tight_layout()
    ax.set_title(f'Moving average of priority-policy buffer size (window size {window_size})')
    ax.set_xlabel('Stream item')
    ax.set_ylabel('Buffer size')

    ax.plot(x, y, label='Priority-policy buffer size')
    ax.hlines(c_mon.total_model_buf_sizes, x[0], x[-1], colors='red', label='Sum of model-buffer sizes')
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.3))

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'pp_buf_size_lineplot.pdf'), format='pdf', bbox_inches='tight')


def create_all_plots_runtime_monitor(rt_mon : RuntimeMonitor, output_dir : Path) -> None:
    """Plots all possible plots to evaluate given RuntimeMonitor."""
    plot_buf_update_rt(rt_mon, output_dir)
    plot_stream_item_processing_rt(rt_mon, output_dir)


def create_all_plots_cache_monitor(c_mon : CacheMonitor, output_dir : Path) -> None:
    """Plots all possible plots to evaluate given CacheMonitor."""
    plot_cache_stats_per_buf_over_stream(c_mon, output_dir)
    plot_cache_stats_per_ot_over_stream(c_mon, output_dir)
    plot_full_cache_evic_over_stream(c_mon, output_dir)

    plot_cache_stats_per_buf_total(c_mon, output_dir)
    plot_cache_stats_per_ot_total(c_mon, output_dir)
    plot_hit_miss_prob(c_mon, output_dir)
    
    plot_ot_frac_per_buf(c_mon, output_dir)
    plot_pp_buf_size_over_stream(c_mon, output_dir)


def get_model_output_path(model_buf : Union[OcdfgBuffer, OcdfgBufferPerObjectType, OcpnBuffer, TotemBuffer]) -> Path:
    """
    Derives path name for monitor-evaluation plots based on given streaming representation.

    Parameters
    ----------
    model_buf : Union[OcdfgBuffer, OcdfgBufferPerObjectType, OcpnBuffer, TotemBuffer]
        Streaming representation whose cache or runtime monitor is evaluated.

    Returns
    -------
    Path
    """
    pp_name = f'{model_buf.pp_buf.prio_order.value.lower()}-{model_buf.pp_buf.pp.value.lower().replace(' ', '-')}' if model_buf.pp_buf is not None else 'none'
    if isinstance(model_buf, OcdfgBuffer):
        model_buf_name = "ocdfg"
        model_buf_sizes = f'{model_buf.node_buf_size}_{model_buf.arc_buf_size}'
    elif isinstance(model_buf, OcdfgBufferPerObjectType):
        model_buf_name = "ocdfg-per-ot"
        model_buf_sizes = f'{model_buf.node_buf_size}_{model_buf.arc_buf_size}'
    elif isinstance(model_buf, OcpnBuffer):
        model_buf_name = "ocpn"
        model_buf_sizes = f'{model_buf.node_buf_size}_{model_buf.arc_buf_size}_{model_buf.ea_buf_size}'
    else:
        model_buf_name = "totem"
        model_buf_sizes = f'{model_buf.tr_buf_size}_{model_buf.ec_buf_size}_{model_buf.lc_buf_size}'

    path_dir = Path(model_buf_name,
                    model_buf_sizes,
                    f"{model_buf.cp.value.lower()}",
                    f"{pp_name}",
                    f"cr-{int(model_buf.coupled_removal)}"
    )
    return path_dir