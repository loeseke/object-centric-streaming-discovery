"""\
Functionality for creating plots to evaluate online TOTeM model against corresponding offline model.
__author__: "Nina Löseke"
"""

from model_buffers import TotemBuffer
from model_builder_totem import TotemModel, get_totem_accuracy, get_totem_avg_scores
from utils import EventStream
from cache_policy_buffers import CachePolicy
from priority_policy_buffers import PrioPolicyOrder, PPBEventsPerObjectType, PPBStridePerObjectType, PPBLifespanPerObjectType, PPBObjectsPerEvent, PPBObjectsPerObjectType, PPBStridePerObject, PPBLifespanPerObject
from utils import EventStream
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from monitor_evaluation import get_model_output_path
from pathlib import Path
import os
from typing import Any
import time


CACHE_POLICIES = [
    CachePolicy.FIFO, 
    CachePolicy.LRU, 
    CachePolicy.LFU, 
    CachePolicy.LFU_DA, 
    CachePolicy.RR
]

PRIORITY_POLICIES_OT = [
    None,
    PPBStridePerObjectType(prio_order=PrioPolicyOrder.MIN, freeze_or_max_idle=pd.Timedelta(days=1)),
    PPBStridePerObjectType(prio_order=PrioPolicyOrder.MAX, freeze_or_max_idle=pd.Timedelta(days=1)),
    PPBLifespanPerObjectType(prio_order=PrioPolicyOrder.MIN, freeze_or_max_idle=pd.Timedelta(days=1)),
    PPBLifespanPerObjectType(prio_order=PrioPolicyOrder.MAX, freeze_or_max_idle=pd.Timedelta(days=1)),
    PPBObjectsPerEvent(prio_order=PrioPolicyOrder.MIN),
    PPBObjectsPerEvent(prio_order=PrioPolicyOrder.MAX),
    PPBObjectsPerObjectType(prio_order=PrioPolicyOrder.MIN),
    PPBObjectsPerObjectType(prio_order=PrioPolicyOrder.MAX),
    PPBEventsPerObjectType(prio_order=PrioPolicyOrder.MIN),
    PPBEventsPerObjectType(prio_order=PrioPolicyOrder.MAX)
]

PRIORITY_POLICIES_OBJ = [
    None,
    PPBStridePerObject(prio_order=PrioPolicyOrder.MIN),
    PPBStridePerObject(prio_order=PrioPolicyOrder.MAX),
    PPBLifespanPerObject(prio_order=PrioPolicyOrder.MIN),
    PPBLifespanPerObject(prio_order=PrioPolicyOrder.MAX)
]   


def plot_heatmap_cp_x_pp(onl_file_path : str, offl_file_path : str, buf_size : int = 100) -> None:
    """
    Computes and visualizes average precision, recall, and accuracy of online TOTeM on stream simulated from given log for all combinations of cache policy and priority policy for given, fixed model-buffer sizes.
    
    Parameters
    ----------
    onl_file_path : str
        Log in JSON format that is converted into object-centric event stream to process.
    offl_file_path : str
        Log in XML format on which TOTeM model is discovered offline for evaluation.
    buf_size : int, default=100
        Size shared by all model buffers for all runs.

    Returns
    -------
    None
    """
    offl_model = TotemModel(offl_file_path)
    event_stream = EventStream(onl_file_path)
    for coupled_rm in [True, False]:
        heatmap_rows_acc = list()
        heatmap_rows_rec = list()
        heatmap_rows_prec = list()

        for pp_buf in PRIORITY_POLICIES_OT:
            pp_name = f'{pp_buf.prio_order.value.lower()} {pp_buf.pp.value}' if pp_buf is not None else 'none'
            pp_dict_acc = {'pp': pp_name}
            pp_dict_prec = {'pp': pp_name}
            pp_dict_rec = {'pp': pp_name}

            for cp in CACHE_POLICIES:
                print(f'Running coupled removal {coupled_rm}, PP {pp_name}, CP {cp.value}...')

                model_buf = TotemBuffer(buf_size, buf_size, buf_size, cp=cp, coupled_removal=coupled_rm, pp_buf=deepcopy(pp_buf))
                model_buf.process_stream(event_stream.stream)
                onl_model = TotemModel(model_buf)
                score_dict = get_totem_avg_scores(offl_model, onl_model)
                
                pp_dict_acc[cp.value] = score_dict['accuracy']
                pp_dict_rec[cp.value] = score_dict['recall']
                pp_dict_prec[cp.value] = score_dict['precision']

            heatmap_rows_acc.append(pp_dict_acc)
            heatmap_rows_rec.append(pp_dict_rec)
            heatmap_rows_prec.append(pp_dict_prec)

        # Create CP x PP heatmap for given coupled-removal setting
        output_dir = Path('../scoring_output',  os.path.basename(os.path.splitext(onl_file_path)[0]), "totem")
        file_name = f'scoring_heatmap_buf-size-{buf_size}_cr-{int(coupled_rm)}.pdf'
        os.makedirs(output_dir, exist_ok=True)

        fig, axs = plt.subplots(1, 3, figsize=(7,4), sharey=True)
        fig.tight_layout()

        for i, (heatmap_rows, metric_name) in enumerate([(heatmap_rows_acc, 'accuracy'), (heatmap_rows_prec, 'precision'), (heatmap_rows_rec, 'recall')]):

            heatmap_df = pd.DataFrame(data=heatmap_rows).set_index('pp')
            sns.heatmap(data=heatmap_df, vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f", annot_kws={"fontsize":8}, linewidth=0.5, cmap='coolwarm_r', ax=axs[i])

            axs[i].set_title(f'Avg. {metric_name}')
            if i == 0:
                axs[i].set_ylabel('Priority policy')
            else:
                axs[i].set_ylabel('')
            axs[i].set_xlabel('Cache policy')
            axs[i].tick_params('y', labelrotation=0)

        fig.savefig(output_dir / file_name, format='pdf', bbox_inches='tight')


def plot_stream_item_processing_time_over_buf_sizes(onl_file_path : str, buf_sizes : list[int]) -> None:
    """
    Plots average processing time per stream item against different buffer sizes (per run, all model buffers are set to the same size).

    Parameters
    ----------
    onl_file_path : str
        Log in JSON format that is converted into object-centric event stream to process.
    buf_sizes : list[int]
        Different buffer sizes to use for all model buffers.

    Returns
    -------
    None
    """
    event_stream = EventStream(onl_file_path)
    num_processed_items = len(event_stream.events + event_stream.o2o_updates)
    time_df_rows = list()

    for buf_size in buf_sizes:
        model_buf_simple = TotemBuffer(buf_size, buf_size, buf_size, CachePolicy.FIFO, coupled_removal=False, pp_buf=None)
        model_buf_mid = TotemBuffer(buf_size, buf_size, buf_size, CachePolicy.LRU, coupled_removal=True, pp_buf=PPBEventsPerObjectType(prio_order=PrioPolicyOrder.MIN))
        model_buf_complex = TotemBuffer(buf_size, buf_size, buf_size, CachePolicy.LFU_DA, coupled_removal=True, pp_buf=PPBStridePerObjectType(prio_order=PrioPolicyOrder.MAX, freeze_or_max_idle=pd.Timedelta(days=1)))

        for i, model_buf in enumerate([model_buf_simple, model_buf_mid, model_buf_complex]):
            print(f'Processing model {i} w/ buffer size {buf_size}...')

            pp_name = f'{model_buf.pp_buf.prio_order.value.lower()} {model_buf.pp_buf.pp.value}' if model_buf.pp_buf is not None else 'none'
            model_buf_name = f'{model_buf.cp.value}, PP {pp_name}, CR {model_buf.coupled_removal}'

            start_time = time.time_ns()
            model_buf.process_stream(event_stream.stream)
            item_processing_time = (time.time_ns() - start_time) / num_processed_items

            time_df_rows.append({'item-processing time': item_processing_time/10**6, 'model-buffer name': model_buf_name, 'buf size': buf_size})

    output_dir = Path('../scoring_output',  os.path.basename(os.path.splitext(onl_file_path)[0]), "totem")
    file_name = f'avg_item_time_over_buf_sizes.pdf'
    os.makedirs(output_dir, exist_ok=True)

    time_df = pd.DataFrame(data=time_df_rows)

    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    fig.tight_layout()

    ax.set_title(f'Avg. processing time over different TOTeM model-buffer sizes')
    ax.set_xlabel('Size of each model buffer')
    ax.set_ylabel('Avg. time per event or O2O update [ms]')
    sns.lineplot(x='buf size', y='item-processing time', hue='model-buffer name', data=time_df, ax=ax)

    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.37))
    ax.tick_params('x', labelrotation=90)

    fig.savefig(output_dir / file_name, format='pdf', bbox_inches='tight')


def plot_avg_score_over_varied_buf_size(onl_file_path : str, offl_file_path : str, fixed_buf_size : int, buf_sizes : list[int], cp : CachePolicy, pp_buf : Any, coupled_removal : bool) -> None:
    """
    Plots evaluation scores against different buffer sizes for one particular model buffer at a time using the given cache and priority policy. The size of all other model buffers is fixed.

    Parameters
    ----------
    onl_file_path : str
        Log in JSON format that is converted into object-centric event stream to process.
    offl_file_path : str
        Log in XML format on which TOTeM model is discovered offline for evaluation.
    fixed_buf_size : int
        Size of model buffers whose size is fixed.
    buf_sizes : list[int]
        Sizes of "varied" model buffer to test.
    cp : CachePolicy
        Cache policy of streaming representation to use for all runs.
    pp_buf : Any
        Piority-policy buffer of streaming representation to use for all runs.
    coupled_removal : bool
        If True, coupled removal is enabled during stream processing for all runs.

    Returns
    -------
    None
    """
    event_stream = EventStream(onl_file_path)
    offl_model = TotemModel(offl_file_path)

    score_rows_acc = list()
    score_rows_prec = list()
    score_rows_rec = list()
    for varied_buf_size in buf_sizes:
        models = list()
        models.append(('TOTeM node buffer', TotemBuffer(varied_buf_size, fixed_buf_size, fixed_buf_size, cp, coupled_removal=coupled_removal, pp_buf=deepcopy(pp_buf))))
        models.append(('TOTeM arc buffer', TotemBuffer(fixed_buf_size, varied_buf_size, fixed_buf_size, cp, coupled_removal=coupled_removal, pp_buf=deepcopy(pp_buf))))
        models.append(('TOTeM event-activity buffer', TotemBuffer(fixed_buf_size, fixed_buf_size, varied_buf_size, cp, coupled_removal=coupled_removal, pp_buf=deepcopy(pp_buf))))
        
        for varied_buf_name, model_buf in models:
            model_buf.process_stream(event_stream.stream)
            onl_model = TotemModel(deepcopy(model_buf))
            score_dict = get_totem_avg_scores(offl_model, onl_model)

            score_rows_acc.append({'varied model buffer': varied_buf_name, 'varied-buffer size': varied_buf_size, 'score': score_dict['accuracy']})
            score_rows_prec.append({'varied model buffer': varied_buf_name, 'varied-buffer size': varied_buf_size, 'score':  score_dict['precision']})
            score_rows_rec.append({'varied model buffer': varied_buf_name, 'varied-buffer size': varied_buf_size, 'score': score_dict['recall']})
    
    # Create plot
    pp_name = f'{pp_buf.prio_order.value.lower()}-{pp_buf.pp.value.lower().replace(' ', '-')}' if pp_buf is not None else 'none'
    output_dir = Path('../scoring_output',  os.path.basename(os.path.splitext(onl_file_path)[0]), "totem")
    file_name = f'scoring_varying_buf-size-{fixed_buf_size}_{model_buf.cp.value.lower()}_{pp_name}_cr-{int(model_buf.coupled_removal)}.pdf'
    os.makedirs(output_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(18,4), sharey=True)
    fig.tight_layout()

    for i, (score_rows, metric_name) in enumerate([(score_rows_acc, 'accuracy'), (score_rows_prec, 'precision'), (score_rows_rec, 'recall')]):
        score_df = pd.DataFrame(data=score_rows)
        bufs = score_df['varied model buffer'].unique()
        buf_colors = sns.color_palette('tab10', len(bufs))
        buf_to_color = dict(zip(bufs, buf_colors))
        sns.lineplot(data=score_df, x='varied-buffer size', y='score', color=buf_to_color, hue='varied model buffer', ax=axs[i])

        axs[i].set_title(f'Avg. {metric_name} for single varied buffer size')
        if i == 0:
            axs[i].set_ylabel('Avg. score')
        else:
            axs[i].set_ylabel('')
        axs[i].set_xlabel('Size of "varied" buffer')
        axs[i].set_ylim((-0.1, 1.1))
        axs[i].get_legend().remove()
    
    lgd_handles = [Line2D([0], [0], color=buf_to_color[buf], label=buf) for buf in buf_to_color]
    fig.legend(loc='lower center', ncol=len(bufs), handles=lgd_handles, bbox_to_anchor=(0.5, -0.15))
    fig.savefig(output_dir / file_name, format='pdf', bbox_inches='tight')


def plot_avg_score_over_buf_sizes(onl_file_path : str, offl_file_path : str, buf_sizes : list[int],  cp : CachePolicy, pp_buf : Any, coupled_removal : bool) -> None:
    """
    Plots average evaluation scores over different buffer sizes for the same stream and streaming representation, i.e. the cache policy, priority policy, and coupled removal are fixed.
    
    Parameters
    ----------
    onl_file_path : str
        Log in JSON format that is converted into object-centric event stream to process.
    offl_file_path : str
        Log in XML format on which TOTeM model is discovered offline for evaluation.
    buf_sizes : list[int]
        Different buffer sizes to use for all model buffers.
    cp : CachePolicy
        Cache policy to use for all runs.
    pp_buf : Any
        Priority policy to use for all runs.
    coupled_removal : bool
        If True, coupled removal is enabled during stream processing for all runs.
    
    Returns
    -------
    None
    """
    event_stream = EventStream(onl_file_path)
    offl_model = TotemModel(offl_file_path)

    score_df_rows = list()
    for buf_size in buf_sizes:
        model_buf = TotemBuffer(buf_size, buf_size, buf_size, cp, coupled_removal=coupled_removal, pp_buf=deepcopy(pp_buf))
        model_buf.process_stream(event_stream.stream)
        onl_model = TotemModel(deepcopy(model_buf))
        score_dict = get_totem_avg_scores(offl_model, onl_model)
        score_dict_annot = get_totem_accuracy(offl_model, onl_model)
        score_dict['TR accuracy'] = score_dict_annot['TR accuracy']
        score_dict['LC accuracy'] = score_dict_annot['LC accuracy']
        score_dict['EC accuracy'] = score_dict_annot['EC accuracy']
        score_dict['buf size'] = buf_size
        score_df_rows.append(score_dict)
        print(score_dict)
    
    # Create plot
    pp_name = f'{pp_buf.prio_order.value.lower()}-{pp_buf.pp.value.lower().replace(' ', '-')}' if pp_buf is not None else 'none'
    output_dir = Path('../scoring_output',  os.path.basename(os.path.splitext(onl_file_path)[0]), "totem")
    file_name = f'scoring_over_buf_sizes_{model_buf.cp.value.lower()}_{pp_name}_cr-{int(model_buf.coupled_removal)}.pdf'
    os.makedirs(output_dir, exist_ok=True)

    score_df = pd.DataFrame(data=score_df_rows)
    score_df = score_df.set_index('buf size')

    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    fig.tight_layout()

    ax.set_title(f'Avg. scores over different model-buffer sizes')
    ax.set_xlabel('Size of each model buffer')
    ax.set_ylabel('Score')
    ax.set_xticks(score_df.index)
    sns.lineplot(data=score_df, ax=ax)

    ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.4))
    ax.tick_params('x', labelrotation=90)

    fig.savefig(output_dir / file_name, format='pdf', bbox_inches='tight')


def plot_score_over_stream(onl_file_path : str, offl_file_path : str, model_buffer : TotemBuffer) -> None:
    """
    Plots evaluation scores over the course of the stream for a given stream and streaming representation.

    Parameters
    ----------
    onl_file_path : str
        Log in JSON format that is converted into object-centric event stream to process.
    offl_file_path : str
        Log in XML format on which TOTeM model is discovered offline for evaluation.
    model_buffer : TotemBuffer
        Streaming-representation object to process stream on.
    
    Returns
    -------
    None
    """
    event_stream = EventStream(onl_file_path)
    offl_model = TotemModel(offl_file_path)

    score_df_rows = list()    
    for pct, stream_chunk in event_stream.create_stream_chunks().items():
        model_buffer.process_stream(stream_chunk)
        onl_model = TotemModel(deepcopy(model_buffer))
        
        score_dict = get_totem_accuracy(offl_model, onl_model)
        score_dict['pct'] = pct
        score_df_rows.append(score_dict)
    
    # Create plot
    output_dir = Path('../scoring_output',  os.path.basename(os.path.splitext(onl_file_path)[0])) / get_model_output_path(model_buffer)
    file_name = 'scoring_over_stream.pdf'
    os.makedirs(output_dir, exist_ok=True)

    score_df = pd.DataFrame(data=score_df_rows)
    score_df = score_df.set_index('pct')

    # Filter DataFrame columns based on metric
    cols_acc = [col for col in score_df.columns if 'accuracy' in col]
    cols_prec = [col for col in score_df.columns if 'precision' in col]
    cols_etc = set(score_df.columns) - set(cols_acc) - set(cols_prec)

    fig, axs = plt.subplots(1, 3, figsize=(18,4), sharex=True, sharey=True)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1)

    for i, (metric, cols) in enumerate([('accuracy', cols_acc), ('precision', cols_prec), ('misc.', cols_etc)]):
        axs[i].set_title(f'Online-vs-offline {metric} scoring over stream')
        axs[i].set_xlabel('Position in stream [%]')
        axs[i].set_ylabel('Score')
        axs[i].set_ylim(-0.1, 1.1)
        axs[i].set_xticks(score_df[list(cols)].index)
        sns.lineplot(data=score_df[list(cols)], ax=axs[i])

        if metric != 'accuracy':
            y_legend = -0.3
        else:
            y_legend = -0.36
        axs[i].legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, y_legend))

    fig.savefig(output_dir / file_name, format='pdf', bbox_inches='tight')