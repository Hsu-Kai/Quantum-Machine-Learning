from ase.db import connect
#from pennylane.templates.embeddings import AmplititudeEmbedding
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP
from dscribe.descriptors import MBTR
from dscribe.descriptors import LMBTR
import glob
import jax
import jax.numpy as jnp
import seaborn as sns
import pandas as pd
import optax  # optimization using jax
import itertools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import json

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



fnames = glob.glob('/work/hsukaicheng/NiCoTiZrHf/train_energies/NiCoTiZrHf.db')


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
            mbtr = MBTR(
                species = list(symbols.keys()),
                geometry={"function": "inverse_distance"},
                grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
                weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
                periodic=False,
                normalization="l2",
            )
            lmbtr = LMBTR(
                species = list(symbols.keys()),
                geometry={"function": "distance"},
                grid={"min": 0, "max": 5, "n": 100, "sigma": 0.1},
                weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
                periodic=True,
                normalization="l2",
            )                        
            training_set.descriptors_acsf = acsf.create(row.toatoms(), verbose=False)
            training_set.descriptors_soap = soap.create(row.toatoms(), verbose=False)
            training_set.descriptors_mbtr = mbtr.create(row.toatoms(), verbose=False)
            training_set.descriptors_lmbtr = lmbtr.create(row.toatoms(), verbose=False)
            training_set.positions = row.toatoms().get_positions()
            training_set.forces = row.toatoms().get_forces()
            training_set.atomic_numbers = row.toatoms().numbers # get atomic numbers
            training_set.symbols_dict = symbols # get atomic types and numbers
            training_set.symbols_array = symbols_array # get atomic symbols
            training_set.total_energy = row.toatoms().get_total_energy()    # energy in eV, 1 Hatree = 27.211 eV 
            training_set.total_energy_per_atom = row.toatoms().get_total_energy()/len(row.toatoms().get_positions())
    print('size of training set: ', len(training_sets))
    Training_Sets_All.append(training_sets)



# specify the number of layers in input-layer structure: n_input_layer
# specify the number of layers in hiden-layer structure: n_hiden_layer
# specify the number of layers in output-layer structure: n_output_layer
# specify hyperparameters DT
n_input_layer = 2
n_hiden_layer = 3
n_output_layer = 2
DT = 3

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

@qml.qnode(device, interface="jax")
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
    return qml.expval((eV_to_hatree)*(qml.PauliZ(0)@qml.PauliZ(4)+qml.PauliZ(0)@qml.PauliZ(8)+qml.PauliZ(4)@qml.PauliZ(8)))


# load bispectrum
#bispectrum = np.load('/work/hsukaicheng/NiCoTiZrHf/train_energies/QCNN_v12/new_bispectrum55_bzero0.npy')


#Get training data: truncate descriptors and energies
D_all = []
E_all = []
for idx in range(len(Training_Sets_All[0])):
    D_all.append(Training_Sets_All[0][idx].descriptors_soap)
    E_all.append(Training_Sets_All[0][idx].total_energy_per_atom)
E_all = np.array(E_all)
n_components = 512
n_bin = 40
descriptors = Training_Set('NiCoTiZrHf')
D_all = descriptors.truncate_x(descriptors.binned_descriptors(n_bin, D_all), n_components, skip_standardization=True)


def load_data(num_train, num_test, rng):
    # subsample train and test split
    train_indices = rng.choice(len(Training_Sets_All[0]), num_train, replace=False)
    test_indices = rng.choice(
        np.setdiff1d(range(len(Training_Sets_All[0])), train_indices), num_test, replace=False
    )

    E_train, E_test = E_all[train_indices], E_all[test_indices]


    D_train, D_test = D_all[train_indices], D_all[test_indices]
    
    return (
        jnp.asarray(D_train.reshape(num_train, -1)),
        jnp.asarray(E_train),
        jnp.asarray(D_test.reshape(num_test, -1)),
        jnp.asarray(E_test),
    )


@jax.jit
def compute_energy_out(weights, features):
    """Computes the output of qdnn"""
    energy_out = lambda weights, features: DNN_energy(weights, features)
    return jax.vmap(energy_out, in_axes=(None, 0), out_axes=0)(
        weights, features
    )

def compute_energy_accuracy(weights, features, E_train):
    """Computes the accuracy over the provided features"""
    E_pred = compute_energy_out(weights, features)
    energy_loss = jnp.sqrt(jnp.average((E_pred - E_train)**2)) 
    return 1.0 - (energy_loss) / jax.numpy.linalg.norm(E_train)


def compute_energy_cost(weights, features, E_train):
    """Computes the energy-cost over the provided features"""
    E_pred = compute_energy_out(weights, features)
    energy_loss = jnp.sqrt(jnp.average((E_pred - E_train)**2)) 
    return energy_loss


def init_weights():
    """Initializes random weights for the QDNN model."""
    weights = pnp.random.normal(loc=0, scale=1,
            size=(num_wires, n_input_layer + n_hiden_layer + n_output_layer), requires_grad=True)
    #weights_last = pnp.random.normal(loc=0, scale=1, size=4 ** 3 - 1, requires_grad=True)
    #return jnp.array(weights), jnp.array(weights_last)
    return jnp.array(weights)

#energy_value_and_grad = jax.jit(jax.value_and_grad(compute_energy_cost, argnums=[0, 1]))
energy_value_and_grad = jax.jit(jax.value_and_grad(compute_energy_cost, argnums=0))


def train_dnn(n_train, n_test, n_epochs):
    # energy or forces (try predicting forces first)
    """
    Args:
        n_train  (int): number of training examples
        n_test   (int): number of test examples
        n_epochs (int): number of training epochs
        desc  (string): displayed string during optimization

    Returns:
        dict: n_train,
        steps,
        train_cost_epochs,
        train_acc_epochs,
        test_cost_epochs,
        test_acc_epochs

    """
    # load data
    x_train, y_train, x_test, y_test = load_data(n_train, n_test, rng)

    # init weights and optimizer
    weights = init_weights()

    # learning rate decay
    cosine_decay_scheduler = optax.cosine_decay_schedule(0.1, decay_steps=n_epochs, alpha=0.95)
    optimizer = optax.adam(learning_rate=cosine_decay_scheduler)
    opt_state = optimizer.init((weights))

    # data containers
    train_cost_epochs, test_cost_epochs, train_acc_epochs, test_acc_epochs, train_out, test_out, E_train, E_test= [], [], [], [], [], [], [], []

    for step in range(n_epochs):
        # Training step with (adam) optimizer

        train_cost, grad_circuit = energy_value_and_grad(weights, x_train, y_train)
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        weights = optax.apply_updates((weights), updates)
        train_energy_out = compute_energy_out(weights, x_train)
        train_out.append(train_energy_out)
        E_train.append(y_train)
        train_cost_epochs.append(train_cost)


        # compute accuracy on training data
        train_acc = compute_energy_accuracy(weights, x_train, y_train)     
        train_acc_epochs.append(train_acc)

        # compute accuracy and cost on testing data
        test_energy_out = compute_energy_out(weights, x_test)
        test_out.append(test_energy_out)
        E_test.append(y_test)
        test_acc = 1.0 - jnp.sqrt((jnp.average((test_energy_out - y_test)**2))) / jax.numpy.linalg.norm(y_test)  
        test_acc_epochs.append(test_acc)     
        test_cost = jnp.sqrt(jnp.average((test_energy_out - y_test)**2))
        test_cost_epochs.append(test_cost)

    final_weights.append(weights)
    #final_weights_last.append(weights_last)

    return dict(
        n_train=[n_train] * n_epochs,
        step=np.arange(1, n_epochs + 1, dtype=int),
        train_cost=train_cost_epochs,
        train_acc=train_acc_epochs,
        test_cost=test_cost_epochs,
        test_acc=test_acc_epochs,
        E_train=E_train,
        E_test=E_test,
        train_out=train_out,
        test_out=test_out
    )

n_test = 100
n_epochs = 1000
n_reps = 10
final_weights = []
def run_iterations(n_train):
    results_df = pd.DataFrame(
        columns=["train_acc", "train_cost", "test_acc", "test_cost", "step", "n_train",
                 "E_train", "E_test", "train_out", "test_out"]
    )

    for _ in range(n_reps):
        results = train_dnn(n_train=n_train, n_test=n_test, n_epochs=n_epochs)
        results_df = pd.concat(
            [results_df, pd.DataFrame.from_dict(results)], axis=0, ignore_index=True
        )

    return results_df


train_sizes = [20, 50, 100, 200, 400, 800]
results_df = run_iterations(n_train=20)
print('training corresponding to size ', 20, ': done')
for n_train in train_sizes[1:]:
    results_df = pd.concat([results_df, run_iterations(n_train=n_train)])
    print('training corresponding to size ', n_train, ': done')

final_weights = np.array(final_weights)
#final_weights_last = np.array(final_weights_last)
results_df.to_pickle("/work/hsukaicheng/NiCoTiZrHf/train_energies/QDNN_v24/v24_03/qdnn_results_df_v24")
np.save("/work/hsukaicheng/NiCoTiZrHf/train_energies/QDNN_v24/v24_03/qdnn_final_weights_v24", final_weights)
#np.save("/work/hsukaicheng/NiCoTiZrHf/train_energies/QDNN_v24/v24_03/qdnn_final_weights_last_v24", final_weights_last)
print('training for all sizes: done')
