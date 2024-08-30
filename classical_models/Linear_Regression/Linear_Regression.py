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
import optax  # optimization using jax
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



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
        D_min = min(list([min(D_all[i]) for i in range(len(D_all))]))
        D_max = max(list([max(D_all[i]) for i in range(len(D_all))]))
        if binned_bispectrum:
            for i in range(len(D_all)):                    
                for channel_id in range(55):
                    histogram, bin_edges = np.histogram(
                    np.array(D_all[i]).reshape(-1,55)[:, channel_id], bins= n_bin, range=(D_min, D_max)
                    )
                    binned_descriptors.append(histogram)
                binned_descriptors_all.append(binned_descriptors)
                binned_descriptors = []
        else:            
            for i in range(len(D_all)):                    
                for channel_id in range(D_all[0].shape[-1]):
                    histogram, bin_edges = np.histogram(
                    D_all[i][:, channel_id], bins= n_bin, range=(D_min, D_max)
                    )
                    binned_descriptors.append(histogram)
                binned_descriptors_all.append(binned_descriptors)
                binned_descriptors = []
        return np.array(binned_descriptors_all)

    def label_binned_descriptors(self, n_bin, D_all):
        c_all = []
        c_per_atom = []
        c_per_training_set = []
        D_min = min(list([D_all[i].min() for i in range(len(D_all))]))
        D_max = max(list([D_all[i].max() for i in range(len(D_all))]))
        bin_edges = np.linspace(np.nextafter(D_min, -np.inf), D_max, n_bin+1)
        for i in range(len(D_all)):
            for k in range(D_all[i].shape[0]):
                for channel_id in range(D_all[0].shape[-1]):
                    c_per_atom.append(next(0.5*(bin_edges[x[0]-1] + bin_edges[x[0]]) for x in enumerate(list(bin_edges)) if x[1] >= D_all[i][k, channel_id]))
                c_per_training_set.append(c_per_atom)
                c_per_atom = []
            c_all.append(c_per_training_set)
            c_per_training_set = []
        return np.array(c_all)
        
    
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
#fnames = glob.glob('/work/hsukaicheng/NiCoTiZrHf/train_energies/NiCoTiZrHf.db')
fnames = glob.glob('/home/hsukaicheng/NiCoTiZrHf/train_energies/NiCoTiZrHf.db')
#fnames = glob.glob('/home/hsukaicheng/PbMOF_BTC_MD/PbMOF_BTC_MD.db')
#fnames = glob.glob('*.db')

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
            #soap_derivatives, soap_descriptors = soap.derivatives(
            #    row.toatoms(),
            #    method="numerical"
            #)
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
            #training_set.soap_derivatives = soap_derivatives[:, :, :, :]
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
    

#Get training data: truncate descriptors and energies
atomic_numbers_arrays = []
D_all, E_all, F_all = [], [], []
for idx in range(len(Training_Sets_All[0])):
	atomic_numbers_arrays.append(Training_Sets_All[0][idx].atomic_numbers)
	D_all.append(Training_Sets_All[0][idx].descriptors_soap)
	E_all.append(Training_Sets_All[0][idx].total_energy_per_atom)
	F_all.append(Training_Sets_All[0][idx].forces)
E_all = np.array(E_all)
F_all = np.array(F_all)
descriptors = Training_Set('NiCoTiZrHf')
D_all = descriptors.descriptors_grouped_by_atoms(atomic_numbers_arrays, D_all)
D_all = D_all.reshape(len(Training_Sets_All[0]), -1)
#D_all = descriptors.truncate_x(D_all, n_components, skip_standardization=False)


num_train = 1500
num_test = 100
train_indices = rng.choice(len(Training_Sets_All[0]), num_train, replace=False)
test_indices = rng.choice(
    np.setdiff1d(range(len(Training_Sets_All[0])), train_indices), num_test, replace=False
)

E_train, E_test = E_all[train_indices], E_all[test_indices]
D_train, D_test = D_all[train_indices], D_all[test_indices]
F_train, F_test = F_all[train_indices], F_all[test_indices]

linear_regression_model = LinearRegression()
linear_regression_model.fit(D_train, E_train)
np.save("linear_regression_model_coef_v00", linear_regression_model.coef_)
np.save("linear_regression_model_intercept_v00", linear_regression_model.intercept_)

y_train_predicted = linear_regression_model.predict(D_train)
y_test_predicted = linear_regression_model.predict(D_test)
rmse_train = np.sqrt(mean_squared_error(E_train, y_train_predicted))
rmse_test = np.sqrt(mean_squared_error(E_test, y_test_predicted))
print(f"RMSE_train: {rmse_train} eV/atom")
print(f"RMSE_test: {rmse_test} eV/atom")
