"""\
Functionality for creating plots to evaluate online OC-DFG or OCPN against corresponding offline model.
__author__: "Nina Löseke"
"""

from model_buffers import OcdfgBuffer, OcdfgBufferPerObjectType, OcpnBuffer, TotemBuffer
from model_builder_ocdfg import OcdfgModel, get_ocdfg_accuracy, get_ocdfg_avg_scores
from model_builder_ocpn import OcpnModel, get_ocpn_accuracy, get_ocpn_avg_scores
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
from typing import Any, Union
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


def plot_heatmap_cp_x_pp(file_path : str, model_buf_name : str, buf_size : int = 100) -> None:
    """
    Computes and visualizes average precision, recall, and accuracy of online model on stream simulated from given log for all combinations of cache policy and priority policy for given, fixed model-buffer sizes.
    
    Parameters
    ----------
    file_path : str
        Log in JSON format that is converted into object-centric event stream to process.
    model_buf_name : str
        Name of streaming representation, either "ocdfg", "ocdfg-per-ot", or "ocpn".
    buf_size : int, default=100
        Size shared by all model buffers of streaming representation for all runs.

    Returns
    -------
    None

    Raises
    ------
    NotImplementedError
        Error thrown if specified `model_buf_name` is not one of "ocdfg", "ocdfg-per-ot", or "ocpn".
    """
    if model_buf_name in ["ocdfg", "ocdfg-per-ot"]:
        offl_model = OcdfgModel(file_path)
    elif model_buf_name == "ocpn":
        offl_model = OcpnModel(file_path)
    else:
        raise NotImplementedError('Unsupported model name! Try "ocdfg", "ocdfg-per-ot", or "ocpn".')

    event_stream = EventStream(file_path)
    for coupled_rm in [True, False]:
        heatmap_rows_acc = list()
        heatmap_rows_rec = list()
        heatmap_rows_prec = list()

        for pp_buf in PRIORITY_POLICIES_OBJ if model_buf_name == "ocdfg-per-ot" else PRIORITY_POLICIES_OT:
            pp_name = f'{pp_buf.prio_order.value.lower()} {pp_buf.pp.value}' if pp_buf is not None else 'none'
            pp_dict_acc = {'pp': pp_name}
            pp_dict_prec = {'pp': pp_name}
            pp_dict_rec = {'pp': pp_name}

            for cp in CACHE_POLICIES:
                print(f'Running coupled removal {coupled_rm}, PP {pp_name}, CP {cp.value}...')

                if model_buf_name == "ocdfg":
                    model_buf = OcdfgBuffer(buf_size, buf_size, cp=cp, coupled_removal=coupled_rm, pp_buf=deepcopy(pp_buf))
                elif model_buf_name == "ocdfg-per-ot":
                    model_buf = OcdfgBufferPerObjectType(buf_size, buf_size, cp=cp, coupled_removal=coupled_rm, pp_buf=deepcopy(pp_buf))
                elif model_buf_name == "ocpn":
                    model_buf = OcpnBuffer(buf_size, buf_size, buf_size, cp=cp, coupled_removal=coupled_rm, pp_buf=deepcopy(pp_buf))
                else:
                    raise NotImplementedError('Unsupport model name! Try "ocdfg", "ocdfg-per-ot", or "ocpn".')
                
                model_buf.process_stream(event_stream.stream)
                if isinstance(model_buf, (OcdfgBuffer, OcdfgBufferPerObjectType)):
                    onl_model = OcdfgModel(model_buf)
                    score_dict = get_ocdfg_avg_scores(offl_model, onl_model)
                else:
                    onl_model = OcpnModel(model_buf)
                    score_dict = get_ocpn_avg_scores(offl_model, onl_model)
                
                pp_dict_acc[cp.value] = score_dict['accuracy']
                pp_dict_rec[cp.value] = score_dict['recall']
                pp_dict_prec[cp.value] = score_dict['precision']

            heatmap_rows_acc.append(pp_dict_acc)
            heatmap_rows_rec.append(pp_dict_rec)
            heatmap_rows_prec.append(pp_dict_prec)

        # Create CP x PP heatmap for given coupled-removal setting
        output_dir = Path('../scoring_output',  os.path.basename(os.path.splitext(file_path)[0]), model_buf_name)
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


def plot_stream_item_processing_time_over_buf_sizes(file_path : str, buf_sizes : list[int]) -> None:
    """
    Plots average processing time per stream item against different buffer sizes (per run, all model buffers are set to the same size).

    Parameters
    ----------
    file_path : str
        Log that is converted into object-centric event stream to process.
    buf_sizes : list[int]
        Different buffer sizes to use for all model buffers.
    
    Returns
    -------
    None
    """
    event_stream = EventStream(file_path)
    num_processed_items = len(event_stream.events)
    time_df_rows = list()
    for buf_size in buf_sizes:
        model_buf_simple = OcdfgBuffer(buf_size, buf_size, CachePolicy.FIFO, coupled_removal=False, pp_buf=None)
        model_buf_mid = OcdfgBuffer(buf_size, buf_size, CachePolicy.LRU, coupled_removal=True, pp_buf=PPBEventsPerObjectType(prio_order=PrioPolicyOrder.MIN))
        model_buf_complex = OcdfgBuffer(buf_size, buf_size, CachePolicy.LFU_DA, coupled_removal=True, pp_buf=PPBStridePerObjectType(prio_order=PrioPolicyOrder.MAX, freeze_or_max_idle=pd.Timedelta(days=1)))

        for model_buf in [model_buf_simple, model_buf_mid, model_buf_complex]:
            pp_name = f'{model_buf.pp_buf.prio_order.value.lower()} {model_buf.pp_buf.pp.value}' if model_buf.pp_buf is not None else 'none'
            model_buf_name = f'{model_buf.cp.value}, PP {pp_name}, CR {model_buf.coupled_removal}'

            start_time = time.time_ns()
            model_buf.process_stream(event_stream.stream)
            ev_processing_time = (time.time_ns() - start_time) / num_processed_items

            time_df_rows.append({'event-processing time': ev_processing_time/10**6, 'model-buffer name': model_buf_name, 'buf size': buf_size})
    
    output_dir = Path('../scoring_output',  os.path.basename(os.path.splitext(file_path)[0]), "ocdfg")
    file_name = f'avg_item_time_over_buf_sizes.pdf'
    os.makedirs(output_dir, exist_ok=True)

    time_df = pd.DataFrame(data=time_df_rows)

    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    fig.tight_layout()

    ax.set_title(f'Avg. event-processing time over different OC-DFG model-buffer sizes')
    ax.set_xlabel('Size of each model buffer')
    ax.set_ylabel('Avg. time per event [ms]')
    sns.lineplot(x='buf size', y='event-processing time', hue='model-buffer name', data=time_df, ax=ax)

    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.37))
    ax.tick_params('x', labelrotation=90)

    fig.savefig(output_dir / file_name, format='pdf', bbox_inches='tight')


def plot_avg_score_over_varied_buf_size(file_path : str, fixed_buf_size : int, buf_sizes : list[int], model_buf_name : str, cp : CachePolicy, pp_buf : Any, coupled_removal : bool, use_mixed_ocdfg_buf : bool = True, pp_buf_dfgs : Any = None) -> None:
    """
    Plots evaluation scores against different buffer sizes for one particular model buffer at a time using the given cache and priority policy. The size of all other model buffers is fixed.

    Parameters
    ----------
    file_path : str
        Log that is converted into object-centric event stream to process.
    fixed_buf_size : int
        Size of model buffers whose size is fixed.
    buf_sizes : list[int]
        Sizes of "varied" model buffer to test.
    model_buf_name : str
        Name of streaming representation, either "ocdfg", "ocdfg-per-ot", or "ocpn".
    cp : CachePolicy
        Cache policy of streaming representation to use for all runs.
    pp_buf : Any
        Piority-policy buffer of streaming representation to use for all runs.
    coupled_removal : bool
        If True, coupled removal is enabled during stream processing for all runs.
    use_mixed_ocdfg_buf : bool, default=True
        If True, OC-DFG model buffers are shared by all object types.
    pp_buf_dfgs : Any, default=None
        Optional priority-policy buffer if model buffers per object type are maintained separately for streaming OC-DFG.

    Returns
    -------
    None

    Raises
    ------
    NotImplementedError
        Error thrown if specified `model_buf_name` is not one of "ocdfg", "ocdfg-per-ot", or "ocpn".
    """
    event_stream = EventStream(file_path)
    if model_buf_name in ["ocdfg", "ocdfg-per-ot"]:
        offl_model = OcdfgModel(file_path)
    elif model_buf_name == "ocpn":
        offl_model = OcpnModel(file_path)
    else:
        raise NotImplementedError('Unsupport model name! Try "ocdfg", "ocdfg-per-ot", or "ocpn".')

    score_rows_acc = list()
    score_rows_prec = list()
    score_rows_rec = list()
    for varied_buf_size in buf_sizes:
        models = list()
        if model_buf_name == "ocdfg":
            models.append(('OC-DFG node buffer', OcdfgBuffer(varied_buf_size, fixed_buf_size, cp, coupled_removal=coupled_removal, pp_buf=deepcopy(pp_buf))))
            models.append(('OC-DFG arc buffer', OcdfgBuffer(fixed_buf_size, varied_buf_size, cp, coupled_removal=coupled_removal, pp_buf=deepcopy(pp_buf))))
        elif model_buf_name == "ocdfg-per-ot":
            models.append(('OC-DFG node buffer', OcdfgBufferPerObjectType(varied_buf_size, fixed_buf_size, cp, coupled_removal=coupled_removal, pp_buf=deepcopy(pp_buf))))
            models.append(('OC-DFG arc buffer', OcdfgBufferPerObjectType(fixed_buf_size, varied_buf_size, cp, coupled_removal=coupled_removal, pp_buf=deepcopy(pp_buf))))
        elif model_buf_name == "ocpn":
            models.append(('OCPN node buffer', OcpnBuffer(varied_buf_size, fixed_buf_size, fixed_buf_size, cp, coupled_removal=coupled_removal, use_mixed_ocdfg_buf=use_mixed_ocdfg_buf, pp_buf=deepcopy(pp_buf), pp_buf_dfgs=pp_buf_dfgs)))
            models.append(('OCPN arc buffer', OcpnBuffer(fixed_buf_size, varied_buf_size, fixed_buf_size, cp, coupled_removal=coupled_removal, use_mixed_ocdfg_buf=use_mixed_ocdfg_buf, pp_buf=deepcopy(pp_buf), pp_buf_dfgs=pp_buf_dfgs)))
            models.append(('OCPN event-activity buffer', OcpnBuffer(fixed_buf_size, fixed_buf_size, varied_buf_size, cp, coupled_removal=coupled_removal, use_mixed_ocdfg_buf=use_mixed_ocdfg_buf, pp_buf=deepcopy(pp_buf), pp_buf_dfgs=pp_buf_dfgs)))
        else:
            raise NotImplementedError('Unsupport model name! Try "ocdfg", "ocdfg-per-ot", or "ocpn".')
        
        for varied_buf_name, model_buf in models:
            model_buf.process_stream(event_stream.stream)
            
            if isinstance(model_buf, (OcdfgBuffer, OcdfgBufferPerObjectType)):
                onl_model = OcdfgModel(deepcopy(model_buf))
                score_dict = get_ocdfg_avg_scores(offl_model, onl_model)
            else:
                onl_model = OcpnModel(deepcopy(model_buf))
                score_dict = get_ocpn_avg_scores(offl_model, onl_model)

            score_rows_acc.append({'varied model buffer': varied_buf_name, 'varied-buffer size': varied_buf_size, 'score': score_dict['accuracy']})
            score_rows_prec.append({'varied model buffer': varied_buf_name, 'varied-buffer size': varied_buf_size, 'score':  score_dict['precision']})
            score_rows_rec.append({'varied model buffer': varied_buf_name, 'varied-buffer size': varied_buf_size, 'score': score_dict['recall']})
    
    # Create plot
    pp_name = f'{pp_buf.prio_order.value.lower()}-{pp_buf.pp.value.lower().replace(' ', '-')}' if pp_buf is not None else 'none'
    output_dir = Path('../scoring_output',  os.path.basename(os.path.splitext(file_path)[0]), model_buf_name)
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


def plot_avg_score_over_buf_sizes(file_path : str, buf_sizes : list[int],  model_buf_name : str, cp : CachePolicy, pp_buf : Any, coupled_removal : bool, use_mixed_ocdfg_buf : bool = True, pp_buf_dfgs : Any = None) -> None:
    """
    Plots average evaluation scores over different buffer sizes for the same stream and streaming representation, i.e. the cache policy, priority policy, and coupled removal are fixed.
    
    Parameters
    ----------
    file_path : str
        Log that is converted into object-centric event stream to process.
    buf_sizes : list[int]
        Different buffer sizes to use for all model buffers.
    model_buf_name : str
        Name of streaming representation to test.
    cp : CachePolicy
        Cache policy to use for all runs.
    pp_buf : Any
        Priority policy to use for all runs.
    coupled_removal : bool
        If True, coupled removal is enabled during stream processing for all runs.
    use_mixed_ocdfg_buf : bool, default=True
        If True, OC-DFG model buffers are shared by all object types.
    pp_buf_dfgs : Any, default=None
        Optional priority-policy buffer if model buffers per obect type are maintained separately for streaming OC-DFG.
    
    Returns
    -------
    None

    Raises
    ------
    NotImplementedError
        Error thrown if specified `model_buf_name` is not one of "ocdfg", "ocdfg-per-ot", or "ocpn".
    """
    event_stream = EventStream(file_path)
    if model_buf_name in ["ocdfg", "ocdfg-per-ot"]:
        offl_model = OcdfgModel(file_path)
    elif model_buf_name == "ocpn":
        offl_model = OcpnModel(file_path)
    else:
        raise NotImplementedError('Unsupport model name! Try "ocdfg", "ocdfg-per-ot", or "ocpn".')

    score_df_rows = list()
    score_df_rows_mae = list()
    for buf_size in buf_sizes:
        if model_buf_name == "ocdfg":
            model_buf = OcdfgBuffer(buf_size, buf_size, cp, coupled_removal=coupled_removal, pp_buf=deepcopy(pp_buf))
        elif model_buf_name == "ocdfg-per-ot":
            model_buf = OcdfgBufferPerObjectType(buf_size, buf_size, cp, coupled_removal=coupled_removal, pp_buf=deepcopy(pp_buf))
        elif model_buf_name == "ocpn":
            model_buf = OcpnBuffer(buf_size, buf_size, buf_size, cp, coupled_removal=coupled_removal, use_mixed_ocdfg_buf=use_mixed_ocdfg_buf, pp_buf=deepcopy(pp_buf), pp_buf_dfgs=pp_buf_dfgs)
        else:
            raise NotImplementedError('Unsupport model name! Try "ocdfg", "ocdfg-per-ot", or "ocpn".')
    
        model_buf.process_stream(event_stream.stream)
        if isinstance(model_buf, (OcdfgBuffer, OcdfgBufferPerObjectType)):
            onl_model = OcdfgModel(deepcopy(model_buf))
            score_dict = get_ocdfg_avg_scores(offl_model, onl_model)
            score_dict_mae = get_ocdfg_accuracy(offl_model, onl_model)
            mae_dict = dict()
            mae_dict['node MAE'] = score_dict_mae['total node freq. MAE']
            mae_dict['arc MAE'] = score_dict_mae['arc freq. MAE']
            mae_dict['buf size'] = buf_size
            score_df_rows_mae.append(mae_dict)
        else:
            onl_model = OcpnModel(deepcopy(model_buf))
            score_dict = get_ocpn_avg_scores(offl_model, onl_model)
        
        score_dict['buf size'] = buf_size
        score_df_rows.append(score_dict)
        print(score_dict)
    
    # Create plot
    pp_name = f'{pp_buf.prio_order.value.lower()}-{pp_buf.pp.value.lower().replace(' ', '-')}' if pp_buf is not None else 'none'
    output_dir = Path('../scoring_output',  os.path.basename(os.path.splitext(file_path)[0]), model_buf_name)
    file_name = f'scoring_over_buf_sizes_{model_buf.cp.value.lower()}_{pp_name}_cr-{int(model_buf.coupled_removal)}.pdf'
    os.makedirs(output_dir, exist_ok=True)

    score_df = pd.DataFrame(data=score_df_rows)
    score_df = score_df.set_index('buf size')

    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    fig.tight_layout()

    ax.set_title(f'Scores over different model-buffer sizes')
    ax.set_xlabel('Size of each model buffer')
    ax.set_ylabel('Score')
    ax.set_xticks(score_df.index)
    sns.lineplot(data=score_df, ax=ax, legend=(model_buf_name == 'ocpn'))
    
    if model_buf_name == 'ocdfg':
        score_df_mae = pd.DataFrame(data=score_df_rows_mae)
        score_df_mae = score_df_mae.set_index('buf size')
        ax2 = ax.twinx()
        sns.lineplot(data=score_df_mae, ax=ax2, palette=sns.color_palette('tab10', 5)[3:], legend=False)

        ax.set_ylabel('Precision/accuracy/recall score')
        ax2.set_ylabel('Mean absolute error')

        color_to_metric = zip(sns.color_palette('tab10', 5), list(score_df.columns) + list(score_df_mae.columns), ['solid', 'dashed', 'dotted', 'solid', 'dashed'], )
        lgd_handles = [Line2D([0], [0], color=color, label=metric_name, linestyle=linestyle) for color, metric_name, linestyle in color_to_metric]
        ax.legend(loc='lower center', ncol=3, handles=lgd_handles, bbox_to_anchor=(0.5, -0.4))
    else:
        ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.3))

    ax.tick_params('x', labelrotation=90)
    fig.savefig(output_dir / file_name, format='pdf', bbox_inches='tight')


def plot_score_over_stream(file_path : str, model_buffer : Union[OcdfgBuffer, OcdfgBufferPerObjectType, OcpnBuffer]) -> None:
    """
    Plots evaluation scores over the course of the stream for a given stream and streaming representation.

    Parameters
    ----------
    file_path : str
        Log that is converted into object-centric event stream to process.
    model_buffer : Union[OcdfgBuffer, OcdfgBufferPerObjectType, OcpnBuffer]
        Streaming-representation object to process stream on.
    
    Returns
    -------
    None
    """
    event_stream = EventStream(file_path)

    if isinstance(model_buffer, (OcdfgBuffer, OcdfgBufferPerObjectType)):
        offl_model = OcdfgModel(file_path)
    else:
        offl_model = OcpnModel(file_path)

    score_df_rows = list()    
    for pct, stream_chunk in event_stream.create_stream_chunks().items():
        model_buffer.process_stream(stream_chunk)
        if isinstance(model_buffer, (OcdfgBuffer, OcdfgBufferPerObjectType)):
            onl_model = OcdfgModel(deepcopy(model_buffer))
            score_dict = get_ocdfg_accuracy(offl_model, onl_model)
        else:
            onl_model = OcpnModel(deepcopy(model_buffer))
            score_dict = get_ocpn_accuracy(offl_model, onl_model)
        
        score_dict['pct'] = pct
        score_df_rows.append(score_dict)
    
    # Create plot
    output_dir = Path('../scoring_output',  os.path.basename(os.path.splitext(file_path)[0])) / get_model_output_path(model_buffer)
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

        if metric != 'misc.':
            y_legend = -0.3
        else:
            y_legend = -0.36
        axs[i].legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, y_legend))

    fig.savefig(output_dir / file_name, format='pdf', bbox_inches='tight')