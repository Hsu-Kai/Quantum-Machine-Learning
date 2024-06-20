from ase.db import connect
#from pennylane.templates.embeddings import AmplititudeEmbedding
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP
import glob
import jax
import jax.numpy as jnp
import seaborn as sns
import pandas as pd
import optax  # optimization using jax
import json


#ref: https://www.nature.com/articles/s43588-021-00084-1

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

seed = 0
rng = np.random.default_rng(seed=seed)


class Training_Set():
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return ('{self.name}'.format(self=self))
    
    def binned_descriptors(self, n_bin, D_all, binned_bispectrum=False):
        binned_descriptors_all = []
        binned_descriptors = []
        if binned_bispectrum:
            D_min = min(list([min(D_all[i]) for i in range(len(D_all))]))
            D_max = max(list([max(D_all[i]) for i in range(len(D_all))]))
            for i in range(len(D_all)):                    
                for channel_id in range(55):
                    histogram, bin_edges = np.histogram(
                    np.array(D_all[i]).reshape(-1,55)[:, channel_id], bins= n_bin, range=(D_min, D_max)
                    )
                    binned_descriptors.append(histogram)
                binned_descriptors_all.append(binned_descriptors)
                binned_descriptors = []
        else:
            D_min = min(list([D_all[i].min() for i in range(len(D_all))]))
            D_max = max(list([D_all[i].max() for i in range(len(D_all))]))
            for i in range(len(D_all)):                    
                for channel_id in range(D_all[0].shape[-1]):
                    histogram, bin_edges = np.histogram(
                    D_all[i][:, channel_id], bins= n_bin, range=(D_min, D_max)
                    )
                    binned_descriptors.append(histogram)
                binned_descriptors_all.append(binned_descriptors)
                binned_descriptors = []
        return np.array(binned_descriptors_all)

    def label_binned_descriptors(self, n_bin, D_all, bispectrum_descriptors=False):
        c_all = []
        c_per_atom = []
        c_per_training_set = []
        if bispectrum_descriptors:
            D_min = min(list([min(D_all[i]) for i in range(len(D_all))]))
            D_max = max(list([max(D_all[i]) for i in range(len(D_all))]))
            bin_edges = np.linspace(np.nextafter(D_min, -np.inf), D_max, n_bin+1)
            for i in range(len(D_all)):
                for k in range(np.array(D_all[i]).reshape(-1,55).shape[0]):
                    for channel_id in range(np.array(D_all[i]).reshape(-1,55).shape[-1]):
                        c_per_atom.append(next(0.5*(bin_edges[x[0]-1] + bin_edges[x[0]]) for x in enumerate(list(bin_edges)) if x[1] >= np.array(D_all[i]).reshape(-1,55)[k, channel_id]))
                    c_per_training_set.append(c_per_atom)
                    c_per_atom = []
                c_all.append(np.array(c_per_training_set))
                c_per_training_set = []
        else:
            D_min = min(list([D_all[i].min() for i in range(len(D_all))]))
            D_max = max(list([D_all[i].max() for i in range(len(D_all))]))
            bin_edges = np.linspace(np.nextafter(D_min, -np.inf), D_max, n_bin+1)
            for i in range(len(D_all)):
                for k in range(D_all[i].shape[0]):
                    for channel_id in range(D_all[i].shape[-1]):
                        c_per_atom.append(next(0.5*(bin_edges[x[0]-1] + bin_edges[x[0]]) for x in enumerate(list(bin_edges)) if x[1] >= D_all[i][k, channel_id]))
                    c_per_training_set.append(c_per_atom)
                    c_per_atom = []
                c_all.append(np.array(c_per_training_set))
                c_per_training_set = []
        return c_all
        
    
    def cumulative_distribution(self, D_all, n = 10, binning=False):
        if binning:
          c_all = []
          c_per_training_set = []
          D_min = min(list([D_all[i].min() for i in range(len(D_all))]))
          D_max = max(list([D_all[i].max() for i in range(len(D_all))]))
          bin_edges = np.linspace(np.nextafter(D_min, -np.inf), D_max, n_bin)
          histograms = self.binned_descriptors(n_bin, D_all)
          cumulative_hist = np.cumsum(histograms, axis = 2)
          n_embeddings = np.linspace(0, 1, n+1)
          for i in range(len(cumulative_hist)):
              for j in range(cumulative_hist[i].shape[0]):
                  cumulative_prob = cumulative_hist[i][j]/cumulative_hist[i][j][-1]
                  cumulative_embeddings = np.interp(n_embeddings[1:], cumulative_prob, bin_edges)
                  c_per_training_set.append(cumulative_embeddings)
              c_all.append(c_per_training_set)
              c_per_training_set = []
        else:        
          c_all = []
          c_per_training_set = []
          n_embeddings = np.linspace(0, 1, n+1)
          for i in range(len(D_all)):
            if D_all[i].shape[0] == Training_Sets_All[0][i].positions.shape[0]:
              D_all[i] = np.swapaxes(D_all[i], 0, 1)
            else:
              pass
            for j in range(D_all[i].shape[0]):
              cumulative_prob = np.cumsum(np.ones_like(D_all[i][j]))/D_all[i].shape[1]
              cumulative_embeddings = np.interp(n_embeddings[1:], cumulative_prob, np.sort(D_all[i][j]))
              c_per_training_set.append(cumulative_embeddings)
            c_all.append(c_per_training_set)
            c_per_training_set = []
        return np.array(c_all)     

    def descriptors_grouped_by_atoms(self, symbols_arrays, D_all):
        grouped_descriptors_all = []
        grouped_descriptors = []
        assert len(symbols_arrays) == len(D_all), (
        "The lengths of the symbols_arrays and D_all do not match!"
        )
        for i in range(len(D_all)):
            assert symbols_arrays[i].shape[0] == D_all[i].shape[0], (
        "The sizes of the symbols_arrays and D_all do not match!"
        )
            for g in np.unique(symbols_arrays[i]):
                ix = np.where(np.array(symbols_arrays[i]) == g)
                grouped_descriptors.append(D_all[i][ix].sum(axis=0))
            grouped_descriptors_all.append(grouped_descriptors)
            grouped_descriptors = []
        return np.array(grouped_descriptors_all)
                    

    def truncate_x(self, x, n_components, skip_standardization=False):
        # shaping and flattening
        if isinstance(x, np.ndarray):
            n = x.shape[0]
            x = x.reshape(n, -1)   
        else:
            n = len(x)
            x = np.array(x).reshape(n, -1)

        if not skip_standardization:
            try:
                # normalization            
                x_mean = x.mean(axis=0)
                x_std = x.std(axis=0)
                x = (x - x_mean)/x_std
            except:
                print("can't standardize input feature ! ")
                pass    # skip standardization

        # truncate
        e_values, e_vectors = np.linalg.eigh(
        np.einsum('ji,jk->ik', x, x))
        return np.einsum('ij,jk->ik', x, e_vectors[:,-n_components:])



#fnames = glob.glob('/work/hsukaicheng/*.db')
#fnames = glob.glob('/work/hsukaicheng/NiCoTiZrHf/train_energies/NiCoTiZrHf.db')
fnames = glob.glob('/home/hsukaicheng/PbMOF_BTC_MD/PbMOF_BTC_MD.db')
#fnames = glob.glob('*.db')
with open("/home/hsukaicheng/PbMOF_BTC_MD/bispectrum_per_atom_PbMOF_BTC_MD", 'r') as fp:
        bispectrum_per_atom=json.load(fp)



eV_to_hatree = 27.211
Training_Sets_All = []


for fname in fnames:
    name = fname.split('.db')[0]
    print('fname: ', fname)

    db = connect(fname)
    training_sets = []

    for i, row in enumerate(db.select()):
        if ('get_positions' and 'get_forces' and 'get_total_energy' in dir(row.toatoms())):
            training_set = Training_Set(name + '_set' + str(i))
            training_sets.append(training_set)
            training_set.data = row.toatoms()
            symbols = {}
            symbols_array = []
            for j in range(len(row.toatoms())):
                symbols[row.toatoms()[j].symbol] = symbols.get(row.toatoms()[j].symbol, 0)+1
                symbols_array.append(row.toatoms()[j].symbol)
            acsf = ACSF(
                species = list(symbols.keys()),
                r_cut = 6.0,
                #g2_params = [[1, 0]],
                g2_params = [[1, 1], [1, 2], [1, 3]],
                #g4_params = [[0.005, 1, 1]],
                g4_params = [[1, 1, 1], [1, 1, -1]],
            )
            soap = SOAP(
                species=list(symbols.keys()),
                periodic=True,
                r_cut = 1.5,
                n_max = 2,
                l_max = 0,
            )
            #mbtr = MBTR(
            #    species = list(symbols.keys()),
            #    geometry={"function": "inverse_distance"},
            #    grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
            #    weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            #    periodic=False,
            #    normalization="l2",
            #)
            #lmbtr = LMBTR(
            #    species = list(symbols.keys()),
            #    geometry={"function": "distance"},
            #    grid={"min": 0, "max": 5, "n": 100, "sigma": 0.1},
            #    weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            #    periodic=True,
            #    normalization="l2",
            #)                        
            training_set.descriptors_acsf = acsf.create(row.toatoms(), verbose=False)
            training_set.descriptors_soap = soap.create(row.toatoms(), verbose=False)
            #training_set.descriptors_mbtr = mbtr.create(row.toatoms(), verbose=False)
            #training_set.descriptors_lmbtr = lmbtr.create(row.toatoms(), verbose=False)
            training_set.positions = row.toatoms().get_positions()
            training_set.forces = row.toatoms().get_forces()
            training_set.atomic_numbers = row.toatoms().numbers # get atomic numbers
            training_set.symbols_dict = symbols # get atomic types and numbers
            training_set.symbols_array = symbols_array # get atomic symbols
            training_set.total_energy = row.toatoms().get_total_energy()    # energy in eV, 1 Hatree = 27.211 eV 
            training_set.total_energy_per_atom = row.toatoms().get_total_energy()/len(row.toatoms().get_positions())
    print('size of training set: ', len(training_sets))
    Training_Sets_All.append(training_sets)



if False:
    def convolutional_layer(weights, wires, skip_first_layer=True):
        """Adds a convolutional layer to a circuit.
        Args:
            weights (np.array): 1D array with 15 weights of the parametrized gates.
            wires (list[int]): Wires where the convolutional layer acts on.
            skip_first_layer (bool): Skips the first two U3 gates of a layer.
        """
        n_wires = len(wires)
        assert n_wires >= 3, "this circuit is too small!"

        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    if indx % 2 == 0 and not skip_first_layer:
                        qml.U3(*weights[:3], wires=[w])
                        qml.U3(*weights[3:6], wires=[wires[indx + 1]])
                    qml.IsingXX(weights[6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[7], wires=[w, wires[indx + 1]])
                    qml.IsingZZ(weights[8], wires=[w, wires[indx + 1]])
                    qml.U3(*weights[9:12], wires=[w])
                    qml.U3(*weights[12:], wires=[wires[indx + 1]])


    def pooling_layer(weights, wires):
        """Adds a pooling layer to a circuit.
        Args:
            weights (np.array): Array with the weights of the conditional U3 gate.
            wires (list[int]): List of wires to apply the pooling layer on.
        """
        n_wires = len(wires)
        assert len(wires) >= 2, "this circuit is too small!"

        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                m_outcome = qml.measure(w)
                qml.cond(m_outcome, qml.U3)(*weights, wires=wires[indx - 1])

    def conv_and_pooling(kernel_weights, n_wires, skip_first_layer=True):
        """Apply both the convolutional and pooling layer."""
        convolutional_layer(kernel_weights[:15], n_wires, skip_first_layer=skip_first_layer)
        pooling_layer(kernel_weights[15:], n_wires)

    def dense_layer(weights, wires):
        """Apply an arbitrary unitary gate to a specified set of wires."""
        qml.ArbitraryUnitary(weights, wires)

    num_wires = 9
    device = qml.device("default.qubit", wires = num_wires)


    @qml.qnode(device) 
    def conv_net_energy(weights, last_layer_weights, features):
        # convolution net for predicting energies
        """Define the QCNN circuit
        Args:
            weights (np.array): Parameters of the convolution and pool layers.
            last_layer_weights (np.array): Parameters of the last dense layer.
            features (np.array): Input data to be embedded using AmplitudEmbedding."""

        layers = weights.shape[1]
        wires = list(range(num_wires))

        # inputs the state input_state
        qml.AmplitudeEmbedding(features=features, wires=wires, pad_with=0.5, normalize=True) 
        qml.Barrier(wires=wires, only_visual=True)

        # adds convolutional and pooling layers
        for j in range(layers):
            conv_and_pooling(weights[:, j], wires, skip_first_layer=(not j == 0))
            wires = wires[::2]
            qml.Barrier(wires=wires, only_visual=True)

        assert last_layer_weights.size == 4 ** (len(wires)) - 1, (
            "The size of the last layer weights vector is incorrect!"
            f" \n Expected {4 ** (len(wires)) - 1}, Given {last_layer_weights.size}"
        )
        dense_layer(last_layer_weights, wires)
        return qml.expval((eV_to_hatree)*qml.PauliZ(0))






# specify the number of layers in input-layer structure: n_input_layer
# specify the number of layers in hiden-layer structure: n_hiden_layer
# specify the number of layers in output-layer structure: n_output_layer
# specify hyperparameters DT
n_input_layer = 2
n_hiden_layer = 3
n_output_layer = 2
DT = 1

# QDNN layer structure: input layer, hiden layer, output layer
# Regarding the hiden layer, the number of layers is governed by one hyper-parameter
def input_layer(weights, wires):
    """Adds an input-layer structure to a circuit.
    Args:
        weights (np.array): the number of weighting factors depends on the number of
        wires and the number of layers in this input-layer structure
    """
    for j in range(n_input_layer):
        if j % 2 == 0:
            for i in wires:
                qml.RX(weights[i, j], wires=i)
        else: 
            for i in wires:
                qml.RZ(weights[i, j], wires=i)


def hiden_layer(weights, wires):
    """Adds a hiden-layer structure to a circuit.
    Args:
        weights (np.array): the number of weighting factors depends on the number of
        wires and the number of layers in this hiden-layer structure but indep of the
        hyperparameter DE
    """     
    for k in range(DT):
        qml.broadcast(qml.CNOT, wires=wires, pattern="ring")
        for j in range(n_hiden_layer):
            if j % 2 == 0:
                for i in wires:
                    qml.RZ(weights[i, (n_input_layer) + j], wires=i)
            else:
                for i in wires:
                    qml.RX(weights[i, (n_input_layer) + j], wires=i)


def output_layer(weights, wires):
    """Adds an output-layer structure to a circuit.
    Args:
        weights (np.array): the number of weighting factors depends on the number of
        wires and the number of layers in this output-layer structure
    """
    qml.broadcast(qml.CNOT, wires=wires, pattern="ring")
    for j in range(n_output_layer):
        if j % 2 == 0:
            for i in wires:
                qml.RX(weights[i, (n_input_layer) + (n_output_layer) + j], wires=i)
        else: 
            for i in wires:
                qml.RZ(weights[i, (n_input_layer) + (n_output_layer) + j], wires=i)


"""
def dense_layer(weights, wires):
    #Apply an arbitrary unitary gate to a specified set of wires.
    qml.ArbitraryUnitary(weights, wires)
"""


# specify total qubits required
num_wires = 9
device = qml.device("default.qubit", wires = num_wires)

@qml.qnode(device)
def DNN_energy(weights, features):
    #deep neural net for predicting energy
    """Define the QCNN circuit
    Args:
        weights (np.array): Parameters of the deep neural net = input layer
        + hiden layer + output layer.
        last_layer_weights (np.array): Parameters of the last dense layer.
        features (np.array): Input data to be embedded using AmplitudEmbedding."""

    wires = list(range(num_wires))

    # inputs the state
    qml.AmplitudeEmbedding(features=features, wires=wires, pad_with=0.5, normalize=True) #add normalization into one argument here?
    qml.Barrier(wires=wires, only_visual=True)

    #adds deep neural net
    input_layer(weights, wires)
    hiden_layer(weights, wires)
    output_layer(weights, wires)
    qml.Barrier(wires=wires, only_visual=True)

    #wires = wires[::2][::2]
    #assert last_layer_weights.size == 4 ** (len(wires)) - 1, (
    #    "The size of the last layer weights vector is incorrect!"
    #    f" \n Expected {4 ** (len(wires)) - 1}, Given {last_layer_weights.size}"
    #)
    #dense_layer(last_layer_weights, wires)
    return qml.expval((eV_to_hatree)*qml.PauliZ(0))



#Get training data: truncate descriptors and energies
atomic_numbers_arrays = []
symbols_arrays = []
D_all = []
E_all = []
for idx in range(len(Training_Sets_All[0])):
	atomic_numbers_arrays.append(Training_Sets_All[0][idx].atomic_numbers)
	symbols_arrays.append(np.array(Training_Sets_All[0][idx].symbols_array))
	#D_all.append(Training_Sets_All[0][idx].descriptors_acsf)
	D_all.append(Training_Sets_All[0][idx].descriptors_soap)
	E_all.append(Training_Sets_All[0][idx].total_energy_per_atom)

bispectra = []
for i in range(len(bispectrum_per_atom)):
	bispectra.append(np.array(bispectrum_per_atom[i]).reshape(-1,55))


number_of_atoms = [len(symbols_arrays[i]) for i in range(len(symbols_arrays))]

E_all = np.array(E_all)
descriptors = Training_Set('PbMOF_BTC_MD')
n_bin = 18
#D_all = descriptors.descriptors_grouped_by_atoms(atomic_numbers_arrays, D_all)


binned1_descriptors = descriptors.binned_descriptors(n_bin, D_all, binned_bispectrum=False)
binned1_bispectrum = descriptors.binned_descriptors(n_bin, bispectrum_per_atom, binned_bispectrum=True)
binned2_descriptors = descriptors.descriptors_grouped_by_atoms(symbols_arrays, D_all)
binned2_bispectrum = descriptors.descriptors_grouped_by_atoms(symbols_arrays, bispectra)

num_train, num_test = 100, 100
train_indices = rng.choice(len(Training_Sets_All[0]), num_train, replace=False)
test_indices = rng.choice(
    np.setdiff1d(range(len(Training_Sets_All[0])), train_indices), num_test, replace=False
)
#D_train, D_test = D_all[train_indices], D_all[test_indices]
#D_train, D_test = D_train.reshape(num_train, -1), D_test.reshape(num_test, -1)


#final_weights = np.load("/home/hsukaicheng/PbMOF_BTC_MD/QCNN_v17/v17_02/qcnn_final_weights_v17.npy")[45]
#final_weights = np.load("/home/hsukaicheng/PbMOF_BTC_MD/QDNN_v17/v17_02/qdnn_final_weights_v17.npy")[46]
#final_weights = np.load("/home/hsukaicheng/PbMOF_BTC_MD/QDNN_v17/v17_01/qdnn_final_weights_v17.npy")[46]
final_weights = np.load("/home/hsukaicheng/PbMOF_BTC_MD/QDNN_v07/v07_07/qdnn_final_weights_v07.npy")[46]
#final_weights_last = np.load("/home/hsukaicheng/PbMOF_BTC_MD/QCNN_v17/v17_02/qcnn_final_weights_last_v17.npy")[45]
#weights, weights_last = pnp.array([final_weights, final_weights_last], requires_grad=True)
weights = pnp.array(final_weights, requires_grad=True)


cfim_func = qml.qinfo.classical_fisher(DNN_energy, argnums=0)
classical_fisher = cfim_func(weights, binned1_bispectrum[0])
print(classical_fisher)
print(classical_fisher.shape)

cFIM = []
#for i in range(len(test_indices)):
#    cFIM.append(cfim_func(weights, D_test[i]))
for i in test_indices:
    cFIM.append(cfim_func(weights, binned1_bispectrum[i]))
print('length of cFIM: ', len(cFIM))
cFIM = np.array(cFIM)
cFIM = np.mean(cFIM, axis = 0)
evalue, evect = np.linalg.eig(cFIM)
tr=cFIM.trace()
n = num_test
const = n/(2*np.pi*np.log(n))
d=cFIM.shape[0]
kappa = const*d/tr
numerator = np.sum(np.log(1+kappa*evalue))
ed = numerator/np.log(const)
norm_ed = ed / d
print('effective dimension: ', ed)
print('normalised effective dimension: ', ed / d)
