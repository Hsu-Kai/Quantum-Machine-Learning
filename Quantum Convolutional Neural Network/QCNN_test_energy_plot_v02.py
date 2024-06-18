from ase.db import connect
#from pennylane.templates.embeddings import AmplititudeEmbedding
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from dscribe.descriptors import ACSF
import glob
import jax
import jax.numpy as jnp
import seaborn as sns
import pandas as pd
import optax  # optimization using jax



#final_rot_weights = np.load("/home/hsukaicheng/NiCoTiZrHf/train_energies/QCNN_v23/v23_03/qcnn_final_rot_weights_v23.npy")
final_weights = np.load("/home/hsukaicheng/NiCoTiZrHf/train_energies/QCNN_v23/v23_03/qcnn_final_weights_v23.npy")
final_weights_last = np.load("/home/hsukaicheng/NiCoTiZrHf/train_energies/QCNN_v23/v23_03/qcnn_final_weights_last_v23.npy")
results_df = pd.read_pickle("/home/hsukaicheng/NiCoTiZrHf/train_energies/QCNN_v23/v23_03/qcnn_results_df_v23")
train_sizes = [20, 50, 100, 200, 400, 800]



if False:
    # aggregate dataframe
    df_agg = results_df.groupby(["n_train", "step"]).agg(["mean", "std"])
    df_agg = df_agg.reset_index()

    sns.set_style('whitegrid')
    colors = sns.color_palette()
    fig, axes = plt.subplots(ncols=3, figsize=(16.5, 5))

    generalization_errors = []

    # plot losses and accuracies
    for i, n_train in enumerate(train_sizes):
        df = df_agg[df_agg.n_train == n_train]

        dfs = [df.train_cost["mean"], df.test_cost["mean"], df.train_acc["mean"], df.test_acc["mean"]]
        lines = ["o-", "x--", "o-", "x--"]
        labels = [fr"$N={n_train}$", None, fr"$N={n_train}$", None]
        axs = [0,0,2,2]
        
        for k in range(4):
            ax = axes[axs[k]]   
            ax.plot(df.step, dfs[k], lines[k], label=labels[k], markevery=10, color=colors[i], alpha=0.8)


        # plot final loss difference
        dif = df[df.step == 1000].test_cost["mean"] - df[df.step == 1000].train_cost["mean"]
        generalization_errors.append(dif)

    # format loss plot
    ax = axes[0]
    ax.set_title('Train and Test Losses', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # format generalization error plot
    ax = axes[1]
    ax.plot(train_sizes, generalization_errors, "o-", label=r"$gen(\alpha)$")
    ax.set_xscale('log')
    ax.set_xticks(train_sizes)
    ax.set_xticklabels(train_sizes)
    ax.set_title(r'Generalization Error $gen(\alpha) = R(\alpha) - \hat{R}_N(\alpha)$', fontsize=14)
    ax.set_xlabel('Training Set Size')

    # format loss plot
    ax = axes[2]
    ax.set_title('Train and Test Accuracies', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')


    legend_elements = [
        mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_sizes)
        ] + [
        mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
        mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
        ]

    axes[0].legend(handles=legend_elements, ncol=3)
    axes[2].legend(handles=legend_elements, ncol=3)

    #axes[1].set_yscale('log', base=2)
    plt.show()



#800 sets for training
z1 = []
z2 = []
for i in np.arange(50999,60000,1000):
	z1.append(results_df["train_out"].iloc[i]-results_df["E_train"].iloc[i])
	z2.append(results_df["test_out"].iloc[i]-results_df["E_test"].iloc[i])
print('800 training sets; training sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z1))**2), axis=1))))
print('800 training sets; testing sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z2))**2), axis=1))))
print('800 training sets; gen err mean: ', np.mean(np.sqrt(np.mean(((np.array(z1))**2), axis=1)))-np.mean(np.sqrt(np.mean(((np.array(z2))**2), axis=1))))
print('800 training sets; training std: ', np.std(np.sqrt(np.mean(((np.array(z1))**2), axis=1))))
print('800 training sets; testing std: ', np.std(np.sqrt(np.mean(((np.array(z2))**2), axis=1))))


#400 sets for training
z3 = []
z4 = []
for i in np.arange(40999,50000,1000):
	z3.append(results_df["train_out"].iloc[i]-results_df["E_train"].iloc[i])
	z4.append(results_df["test_out"].iloc[i]-results_df["E_test"].iloc[i])
print('400 training sets; training sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z3))**2), axis=1))))
print('400 training sets; testing sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z4))**2), axis=1))))
print('400 training sets; gen err mean: ', np.mean(np.sqrt(np.mean(((np.array(z3))**2), axis=1)))-np.mean(np.sqrt(np.mean(((np.array(z4))**2), axis=1))))
print('400 training sets; training std: ', np.std(np.sqrt(np.mean(((np.array(z3))**2), axis=1))))
print('400 training sets; testing std: ', np.std(np.sqrt(np.mean(((np.array(z4))**2), axis=1))))


#200 sets for training
z5=[]
z6=[]
for i in np.arange(30999,40000,1000):
	z5.append(results_df["train_out"].iloc[i]-results_df["E_train"].iloc[i])
	z6.append(results_df["test_out"].iloc[i]-results_df["E_test"].iloc[i])
print('200 training sets; training sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z5))**2), axis=1))))
print('200 training sets; testing sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z6))**2), axis=1))))
print('200 training sets; gen err mean: ', np.mean(np.sqrt(np.mean(((np.array(z5))**2), axis=1)))-np.mean(np.sqrt(np.mean(((np.array(z6))**2), axis=1))))
print('200 training sets; training std: ', np.std(np.sqrt(np.mean(((np.array(z5))**2), axis=1))))
print('200 training sets; testing std: ', np.std(np.sqrt(np.mean(((np.array(z6))**2), axis=1))))



#100 sets for training
z7=[]
z8=[]
for i in np.arange(20999,30000,1000):
	z7.append(results_df["train_out"].iloc[i]-results_df["E_train"].iloc[i])
	z8.append(results_df["test_out"].iloc[i]-results_df["E_test"].iloc[i])
print('100 training sets; training sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z7))**2), axis=1))))
print('100 training sets; testing sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z8))**2), axis=1))))
print('100 training sets; gen err mean: ', np.mean(np.sqrt(np.mean(((np.array(z7))**2), axis=1)))-np.mean(np.sqrt(np.mean(((np.array(z8))**2), axis=1))))
print('100 training sets; training std: ', np.std(np.sqrt(np.mean(((np.array(z7))**2), axis=1))))
print('100 training sets; testing std: ', np.std(np.sqrt(np.mean(((np.array(z8))**2), axis=1))))



#50 sets for training
z9=[]
z10=[]
for i in np.arange(10999,20000,1000):
	z9.append(results_df["train_out"].iloc[i]-results_df["E_train"].iloc[i])
	z10.append(results_df["test_out"].iloc[i]-results_df["E_test"].iloc[i])
print('50 training sets; training sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z9))**2), axis=1))))
print('50 training sets; testing sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z10))**2), axis=1))))
print('50 training sets; gen err mean: ', np.mean(np.sqrt(np.mean(((np.array(z9))**2), axis=1)))-np.mean(np.sqrt(np.mean(((np.array(z10))**2), axis=1))))
print('50 training sets; training std: ', np.std(np.sqrt(np.mean(((np.array(z9))**2), axis=1))))
print('50 training sets; testing std: ', np.std(np.sqrt(np.mean(((np.array(z10))**2), axis=1))))



#20 sets for training
z11=[]
z12=[]
for i in np.arange(999,10000,1000):
	z11.append(results_df["train_out"].iloc[i]-results_df["E_train"].iloc[i])
	z12.append(results_df["test_out"].iloc[i]-results_df["E_test"].iloc[i])
print('20 training sets; training sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z11))**2), axis=1))))
print('20 training sets; testing sqrt: ', np.mean(np.sqrt(np.mean(((np.array(z12))**2), axis=1))))
print('20 training sets; gen err mean: ', np.mean(np.sqrt(np.mean(((np.array(z11))**2), axis=1)))-np.mean(np.sqrt(np.mean(((np.array(z12))**2), axis=1))))
print('20 training sets; training std: ', np.std(np.sqrt(np.mean(((np.array(z11))**2), axis=1))))
print('20 training sets; testing std: ', np.std(np.sqrt(np.mean(((np.array(z12))**2), axis=1))))



if True:
    x1=results_df["train_out"].iloc[59999]
    y1=results_df["E_train"].iloc[59999]
    x2=results_df["test_out"].iloc[59999]
    y2=results_df["E_test"].iloc[59999]
    #x1=x1*27.211
    #x2=x2*27.211
    #y1=y1*27.211
    #y2=y2*27.211
    fig, axs = plt.subplots(1,1)
    axs.scatter(y1, x1)
    axs.scatter(y2, x2, color = 'red')
    #axs.set_aspect('equal')
    plt.plot(x1,x1, color = 'black')
    axs.legend(['Training Sets', 'Testing Sets'], fontsize="16")
    #axs.set_xlabel(r'$\bf{E_{DFT}\ (eV/atom)}$ ', fontweight ='bold', fontsize="15")
    #axs.set_ylabel(r'$\bf{E_{QMLM }\ (eV/atom)}$ ', fontweight ='bold', fontsize="15")
    axs.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()








