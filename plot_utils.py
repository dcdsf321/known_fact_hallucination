
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset




def plot_result(prob,llama_tokens_str,ans_str,start_tok=0,end_tok=0,rev=False):
    plt.cla()
    plt.clf() 
    tok_len = len(llama_tokens_str)
    # Define a custom colormap with a white to blue gradient
    if rev:
        cmap = plt.get_cmap('Blues_r')
    else:
        cmap = plt.get_cmap('Blues')


    # Create a heatmap
    # fig, ax = plt.subplots(figsize=(12, 5))
    fig, ax = plt.subplots()
    # Plot max_norm in the first subplot
    ax.imshow(prob.T.cpu()[start_tok:tok_len-end_tok,:], cmap=cmap)#vmin=-abs(prob).max(), vmax=abs(prob).max())#, origin='lower')
    # ax.imshow(prob.T.cpu()[start_tok:tok_len-end_tok,:], cmap=cmap2, vmin=0, vmax=1)#, origin='lower')
    ax.set_yticks(np.arange(tok_len-start_tok-end_tok))
    # ax.set_yticklabels(list(reversed(llama_tokens_str[start_tok:])), rotation=45)
    ax.set_yticklabels(llama_tokens_str[start_tok:tok_len-end_tok])
    ax.set_title(f"{ans_str} probablities")
    ax.set_xlabel("layers")
    ax.set_ylabel("tokens")
    ax.set_aspect('auto')  # Adjust aspect ratio if needed
    
    # Add a colorbar for each subplot
    cbar = ax.figure.colorbar(ax.images[0], ax=ax)
    cbar.set_label("Probability", rotation=-90, va="bottom")

    plt.rcParams['axes.grid'] = False 
    plt.rcParams['figure.autolayout'] = True  
    # Show the plots
    plt.tight_layout()
    plt.show()
    
def plot_la_result(la,llama_tokens_str,start_tok=1,end_tok=0,norm_type='max'):
    tok_len = len(llama_tokens_str)
    la_cpu = la.cpu().numpy()
    if norm_type == 'max':
        norm = np.max(la_cpu, axis=2)
    elif norm_type == 'l2':
        norm = np.linalg.norm(la_cpu, axis=2)
    norm = norm.T
    
    # Define a custom colormap with a white to blue gradient
    cmap = plt.get_cmap('Blues')

    # Create a heatmap
    # fig, ax = plt.subplots(figsize=(12, 5))
    fig, ax = plt.subplots()
    # Plot max_norm in the first subplot
    ax.imshow(norm[start_tok:tok_len-end_tok,:], cmap=cmap)#, origin='lower')
    ax.set_yticks(np.arange(tok_len-start_tok-end_tok))
    # ax.set_yticklabels(list(reversed(llama_tokens_str[start_tok:])), rotation=45)
    ax.set_yticklabels(llama_tokens_str[start_tok:tok_len-end_tok])
    ax.set_title(f"{norm_type} norm attn T to i across L")
    ax.set_xlabel("layers")
    ax.set_ylabel("tokens")
    ax.set_aspect('auto')  # Adjust aspect ratio if needed
    
    # Add a colorbar for each subplot
    cbar = ax.figure.colorbar(ax.images[0], ax=ax)
    cbar.set_label("Max Norm")

    # Show the plots
    plt.tight_layout()
    plt.show()


def visualize_heatmap(data,
                      xlabel="layer",
                      ylabel="layer",
                      title=None,
                      fig_dir="figs5",
                      fig_size=None,
                      adjust=None,
                      vmin=None,
                      vmax=None,
                      sort_index=True):
    sns.set_theme()

    if sort_index:
        data.sort_index(ascending=False, inplace=True)

    # if not os.path.exists(f"{fig_dir}/metadata"):
    #     os.makedirs(f"{fig_dir}/metadata")
    # data.to_csv(f"{fig_dir}/metadata/{fname}_heatmap.csv")

    ax = None
    if fig_size is not None:
        fig, ax = plt.subplots(figsize=fig_size)

    sns.heatmap(data, ax=ax,vmin=vmin,vmax=vmax)

    if title is not None:
        plt.title(title)

    if adjust is not None:
        plt.subplots_adjust(bottom=adjust[0], left=adjust[1])

    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)

    # plt.savefig(f"{fig_dir}/{fname}_heatmap.png")
    # plt.savefig(f'{fig_dir}/{fname}_heatmap.pdf', format='pdf')
    plt.show()


def prepare_data_for_heatmap(layer2layer):
    columns = list(layer2layer.keys())
    print(columns)
    columns.remove("layer")
    l2_df = pd.DataFrame(data=layer2layer,
                         columns=columns,
                         index=layer2layer["layer"])

    visualize_heatmap(l2_df)
    
def lines(data_dict, xlabel="layers", ylabel="tokens",title='sub_prob',rev = False,wo_keys = [], filepth = None,vmin=None, vmax=None, **kwargs):
    plt.cla()
    plt.clf() 
    for key, value in data_dict.items():
        if key in wo_keys:
            continue
        plt.plot(value, label=key)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rev:
        plt.gca().invert_yaxis()
    plt.ylim(vmin, vmax)
    plt.title(title) 
    if filepth is not None:
        plt.savefig(filepth)
    else:
        plt.show()
        
    plt.close() 

def heatmap(data_dict, xlabel="layers", ylabel="tokens",title='sub_prob',wo_keys = [], filepth = None, rev = False, vmin=None, vmax=None, **kwargs):
    plt.cla()
    plt.clf() 
    data_list = []
    data_keys = []
    for key, value in data_dict.items():
        if key in wo_keys:
            continue
        data_list.append(value)
        data_keys.append(key)
    data_matrix = np.array(data_list)
    if rev:
        cmap = plt.get_cmap('Blues_r')
    else:
        cmap = plt.get_cmap('Blues')
    plt.imshow(data_matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.yticks(np.arange(len(data_keys)), data_keys)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.colorbar(label='prob')
    plt.tight_layout()
    if filepth is not None:
        plt.savefig(filepth)
    else:
        plt.show()
    plt.close() 
    
