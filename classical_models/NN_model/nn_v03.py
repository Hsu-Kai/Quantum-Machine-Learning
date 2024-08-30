from ase.db import connect
#from pennylane.templates.embeddings import AmplititudeEmbedding
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP
from dscribe.descriptors import CoulombMatrix
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
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

    def descriptor_derivatives_grouped_by_atoms(self, symbols_arrays, D_all):
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
            grouped_descriptors_all.append(np.array(grouped_descriptors))
            grouped_descriptors = []
        return grouped_descriptors_all                    

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
fnames = glob.glob('/work/hsukaicheng/NiCoTiZrHf/train_energies/NiCoTiZrHf.db')
#fnames = glob.glob('/home/hsukaicheng/NiCoTiZrHf/train_energies/NiCoTiZrHf.db')    
#fnames = glob.glob('/home/hsukaicheng/PbMOF_BTC_MD/PbMOF_BTC_MD.db')
#fnames = glob.glob('/work/hsukaicheng/PbMOF_BTC_MD.db')
#fnames = glob.glob('*.db')
#with open("/work/hsukaicheng/PbMOF_BTC_MD/bispectrum_per_atom_PbMOF_BTC_MD", 'r') as fp:
#        bispectrum_per_atom=json.load(fp)




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
            soap_derivatives, soap_descriptors = soap.derivatives(
                row.toatoms(),
                method="numerical"
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
            #cm = CoulombMatrix(n_atoms_max=130)
            training_set.descriptors_acsf = acsf.create(row.toatoms(), verbose=False)
            training_set.descriptors_soap = soap.create(row.toatoms(), verbose=False)
            training_set.soap_derivatives = soap_derivatives[:, :, :, :]
            #training_set.descriptors_mbtr = mbtr.create(row.toatoms(), verbose=False)
            #training_set.descriptors_lmbtr = lmbtr.create(row.toatoms(), verbose=False)
            #training_set.descriptors_coulomb_matrices = cm.create(row.toatoms(), verbose=False)
            training_set.positions = row.toatoms().get_positions()
            training_set.forces = row.toatoms().get_forces()
            training_set.atomic_numbers = row.toatoms().numbers # get atomic numbers
            training_set.symbols_dict = symbols # get atomic types and numbers
            training_set.symbols_array = symbols_array # get atomic symbols
            training_set.total_energy = row.toatoms().get_total_energy()    # energy in eV, 1 Hatree = 27.211 eV 
            training_set.total_energy_per_atom = row.toatoms().get_total_energy()/len(row.toatoms().get_positions())
    print('size of training set: ', len(training_sets))
    Training_Sets_All.append(training_sets)


torch.manual_seed(7)


atomic_numbers_arrays, D_all, E_all, F_all, soap_derivatives = [], [], [], [], []
for idx in range(len(Training_Sets_All[0])):
    atomic_numbers_arrays.append(Training_Sets_All[0][idx].atomic_numbers)
    D_all.append(Training_Sets_All[0][idx].descriptors_soap)
    E_all.append(Training_Sets_All[0][idx].total_energy_per_atom)
    F_all.append(Training_Sets_All[0][idx].forces)
    soap_derivatives.append(Training_Sets_All[0][idx].soap_derivatives)
descriptors = Training_Set('NiCoTiZrHf')
D_all = descriptors.descriptors_grouped_by_atoms(atomic_numbers_arrays, D_all)
D_all = np.array(D_all).reshape(len(D_all), -1)
E_all = np.array(E_all)
F_all = np.array(F_all)
soap_derivatives = descriptors.descriptor_derivatives_grouped_by_atoms(atomic_numbers_arrays, soap_derivatives)
dD_dr_all = []
for i in range(len(soap_derivatives)):
    dD_dr_all.append(soap_derivatives[i].reshape(soap_derivatives[i].shape[1], soap_derivatives[i].shape[2], -1))	
dD_dr_all = np.array(dD_dr_all)
n_samples, n_features = D_all.shape


num_train = 1500
num_test = 100
train_indices = rng.choice(len(Training_Sets_All[0]), num_train, replace=False)
test_indices = rng.choice(
    np.setdiff1d(range(len(Training_Sets_All[0])), train_indices), num_test, replace=False
)
D_train_full, D_test = D_all[train_indices], D_all[test_indices]
E_train_full, E_test = E_all[train_indices], E_all[test_indices]
F_train_full, F_test = F_all[train_indices], F_all[test_indices]
dD_dr_train_full, dD_dr_test = dD_dr_all[train_indices], dD_dr_all[test_indices]
D_train, D_valid, E_train, E_valid, F_train, F_valid, dD_dr_train, dD_dr_valid = train_test_split(
    D_train_full,
    E_train_full,
    F_train_full,
    dD_dr_train_full,
    test_size=0.2,
    random_state=7,
)


"""
# Standardize input for improved learning. Fit is done only on training data,
# scaling is applied to both descriptors and their derivatives on training and
# test sets.
scaler = StandardScaler().fit(D_train_full)
D_train_full = scaler.transform(D_train_full)
D_whole = scaler.transform(D_all)
dD_dr_whole = dD_dr_all / scaler.scale_[None, None, None, :]
dD_dr_train_full = dD_dr_train_full / scaler.scale_[None, None, None, :]

# Calculate the variance of energy and force values for the training set. These
# are used to balance their contribution to the MSE loss
var_energy_train = E_train_full.var()
var_force_train = F_train_full.var()
"""


# Create tensors for pytorch
#D_whole = torch.Tensor(D_whole)
D_train = torch.Tensor(D_train)
D_valid = torch.Tensor(D_valid)
D_test = torch.Tensor(D_test)
E_train = torch.Tensor(E_train)
E_valid = torch.Tensor(E_valid)
E_test = torch.Tensor(E_test)
F_train = torch.Tensor(F_train)
F_valid = torch.Tensor(F_valid)
F_test = torch.Tensor(F_test)
dD_dr_train = torch.Tensor(dD_dr_train)
dD_dr_valid = torch.Tensor(dD_dr_valid)
dD_dr_test = torch.Tensor(dD_dr_test)



# define our model and loss function:

class FFNet(torch.nn.Module):
    """A simple feed-forward network with one hidden layer, randomly
    initialized weights, sigmoid activation and a linear output layer.
    """
    def __init__(self, n_features, n_hidden, n_out):
        super(FFNet, self).__init__()
        self.linear1 = torch.nn.Linear(n_features, n_hidden)
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=1.0)
        self.sigmoid = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(n_hidden, n_out)
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=1.0)
        self.linear = torch.nn.Linear(n_features, n_out)

    def forward(self, x):
        #x = self.linear1(x)
        #x = self.sigmoid(x)
        #x = self.linear2(x)
        x = self.linear(x)

        return x


def energy_loss(E_pred, E_train):
    #Custom loss function that targets energies.
    #energy_loss = torch.mean((E_pred.flatten() - E_train)**2) / var_energy_train
    energy_loss = torch.mean((E_pred.flatten() - E_train)**2)  # energies without scaling
    return energy_loss

"""
def force_loss(F_pred, F_train):
    force_loss = torch.mean((F_pred.flatten() - F_train.flatten())**2) # without scaling
    return force_loss


def energy_force_loss(E_pred, E_train, F_pred, F_train):
    #Custom loss function that targets both energies and forces.
    energy_loss = torch.mean((E_pred - E_train)**2) / var_energy_train
    force_loss = torch.mean((F_pred - F_train)**2) / var_force_train
    return energy_loss + force_loss
"""
    

# Initialize model
model = FFNet(n_features, n_hidden=0, n_out=1)
weights = np.load('/work/hsukaicheng/NiCoTiZrHf/train_energies/classical_model/v03/linear_regression_model_coef_v00.npy')
bias = np.load('/work/hsukaicheng/NiCoTiZrHf/train_energies/classical_model/v03/linear_regression_model_intercept_v00.npy')
para = model.state_dict()
para['linear.weight'] = torch.Tensor(weights).unsqueeze(0)
para['linear.bias'] = torch.Tensor(bias).unsqueeze(0)
model.load_state_dict(para)

# The Adam optimizer is used for training the model parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Train!
n_max_epochs = 300000
batch_size = num_train
patience = 300
i_worse = 0
old_valid_loss = float("Inf")
best_valid_loss = float("Inf")

# We explicitly require that the gradients should be calculated for the input
# variables. PyTorch will not do this by default as it is typically not needed.
D_valid.requires_grad = True

# Epochs
for i_epoch in range(n_max_epochs):

    D_train.requires_grad = True

    #\ Forward pass: Predict energies from the descriptor input
    E_train_pred = model(D_train)

    # Get derivatives of model output with respect to input variables. The
    # torch.autograd.grad-function can be used for this, as it returns the
    # gradients of the input with respect to outputs. It is very important
    # to set the create_graph=True in this case. Without it the derivatives
    # of the NN parameters with respect to the loss from the force error
    # will not be populated (=the force error will not affect the
    # training), but the model will still run fine without errors.

    df_dD_train = torch.autograd.grad(
        outputs=E_train_pred,
        inputs=D_train,
        grad_outputs=torch.ones_like(E_train_pred),
        create_graph=True,
    )[0]

    # Get derivatives of input variables (=descriptor) with respect to atom
    # positions = forces
    F_train_pred = -torch.einsum('ijkl,il->ijk', dD_dr_train, df_dD_train)

    # Zero gradients, perform a backward pass, and update the weights.
    #D_train_batch.grad.data.zero_()
    optimizer.zero_grad()
    loss = energy_loss(E_train_pred, E_train)
    #loss = force_loss(F_train_pred, F_train)
    loss.backward()
    optimizer.step()

    # Check early stopping criterion and save best model
    E_valid_pred = model(D_valid)
    df_dD_valid = torch.autograd.grad(
        outputs=E_valid_pred,
        inputs=D_valid,
        grad_outputs=torch.ones_like(E_valid_pred),
    )[0]
    F_valid_pred = -torch.einsum('ijkl,il->ijk', dD_dr_valid, df_dD_valid)
    valid_loss = energy_loss(E_valid_pred, E_valid)
    #valid_loss = force_loss(F_valid_pred, F_valid)
    if valid_loss < best_valid_loss:
        # print("Saving at epoch {}".format(i_epoch))
        torch.save(model.state_dict(), "best_model.pt")
        best_valid_loss = valid_loss
    if valid_loss >= old_valid_loss:
        i_worse += 1
    else:
        i_worse = 0
    if i_worse > patience:
        print("Early stopping at epoch {}".format(i_epoch))
        break
    old_valid_loss = valid_loss

    if i_epoch % 500 == 0:
        print("  Finished epoch: {} with loss: {}".format(i_epoch, loss.item()))


# Way to tell pytorch that we are entering the evaluation phase
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Calculate energies and force for the entire range
#E_whole = torch.Tensor(E_numpy)
#F_whole = torch.Tensor(F_numpy)
#dD_dr_whole = torch.Tensor(dD_dr_whole)
D_test.requires_grad = True
E_test_pred = model(D_test)
E_train_pred = model(D_train)
E_valid_pred = model(D_valid)
#df_dD_whole = torch.autograd.grad(
#    outputs=E_whole_pred,
#    inputs=D_whole,
#    grad_outputs=torch.ones_like(E_whole_pred),
#)[0]
#F_whole_pred = -torch.einsum('ijkl,il->ijk', dD_dr_whole, df_dD_whole)

df_dD_test = torch.autograd.grad(
    outputs=E_test_pred,
    inputs=D_test,
    grad_outputs=torch.ones_like(E_test_pred),
)[0]
F_test_pred = -torch.einsum('ijkl,il->ijk', dD_dr_test, df_dD_test)

df_dD_train = torch.autograd.grad(
    outputs=E_train_pred,
    inputs=D_train,
    grad_outputs=torch.ones_like(E_train_pred),
)[0]
F_train_pred = -torch.einsum('ijkl,il->ijk', dD_dr_train, df_dD_train)

df_dD_valid = torch.autograd.grad(
    outputs=E_valid_pred,
    inputs=D_valid,
    grad_outputs=torch.ones_like(E_valid_pred),
)[0]
F_valid_pred = -torch.einsum('ijkl,il->ijk', dD_dr_valid, df_dD_valid)


E_train_pred = E_train_pred.detach().numpy()
E_valid_pred = E_valid_pred.detach().numpy()
E_test_pred = E_test_pred.detach().numpy()
F_train_pred = F_train_pred.detach().numpy()
F_valid_pred = F_valid_pred.detach().numpy()
F_test_pred = F_test_pred.detach().numpy()

#E_whole = E_whole.detach().numpy()
E_train = E_train.detach().numpy()
E_valid = E_valid.detach().numpy()
F_train = F_train.detach().numpy()
F_valid = F_valid.detach().numpy()


# Save results for later analysis
np.save("E_train.npy", E_train)
np.save("E_valid.npy", E_valid)
np.save("E_test.npy", E_test)
np.save("F_train.npy", F_train)
np.save("F_valid.npy", F_valid)
np.save("F_test.npy", F_test)

#np.save("F_train_full.npy", F_train_full)
np.save("E_train_pred.npy", E_train_pred)
np.save("E_valid_pred.npy", E_valid_pred)
np.save("E_test_pred.npy", E_test_pred)
np.save("F_train_pred.npy", F_train_pred)
np.save("F_valid_pred.npy", F_valid_pred)
np.save("F_test_pred.npy", F_test_pred)

#np.save("F_whole_pred.npy", F_whole_pred)
