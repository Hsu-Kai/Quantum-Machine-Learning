from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pennylane as qml
import pandas as pd
from sklearn.svm import SVC


# ref: https://pennylane.ai/qml/demos/tutorial_kernels_module/


seed = 0
rng = np.random.default_rng(seed=seed)


class Training_Set():
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return ('{self.name}'.format(self=self))
    
    def label_Tensile_Strength(self, n_bin, D_all):
        "assign classification labels"
        #D_all = data['Tensile Strength (GPa)']
        D_all = D_all[~np.isnan(D_all)]
        clabel= []
        D_min = min(D_all)
        D_max = max(D_all)
        bin_edges = np.linspace(np.nextafter(D_min, -np.inf), D_max, n_bin+1)
        for i in range(len(D_all)):
                    clabel.append(next(x[0] for x in enumerate(list(bin_edges)) if x[1] >= D_all[i]))
        return np.array(clabel)	

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




data = pd.read_csv("/work/hsukaicheng/carbon fiber/GPT-3/GPT_forward_trainingset1.csv")
cols = list(data.columns)
col_num = [1,3,4,5,6,7,8,9,10,11,12,13,14,15,19,20,21,22,23,24,25,26,27]
cols_selected = [cols[i] for i in col_num]

print('number of columns selected: ', len(cols_selected))


### Get training data: truncate descriptors and Tensile Stress

df = data[cols_selected][:179]
#df = data[:179]
TS_all = data['Tensile Strength (GPa)'][:179]
n_components = 16
n_bin = 4
descriptors = Training_Set('carbon_fiber')
#D_all = descriptors.truncate_x(df, n_components, skip_standardization=True)

X = np.array(df)
Y = descriptors.label_Tensile_Strength(n_bin, TS_all)


num_train = 159
num_test = 20
train_indices = rng.choice(len(df), num_train, replace=False)
test_indices = rng.choice(
    np.setdiff1d(range(len(df)), train_indices), num_test, replace=False
)

Y_train, Y_test = Y[train_indices], Y[test_indices]
X_train, X_test = X[train_indices], X[test_indices]


##############################################################################
# Defining a Quantum Embedding Kernel


def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])


##############################################################################
# To construct the ansatz, this layer is repeated multiple times, reusing
# the datapoint x but feeding different variational parameters into each of them.


def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)


##############################################################################
# Together with the ansatz we only need a device to run the quantum circuit on.


dev = qml.device("default.qubit", wires=5, shots=None)
wires = dev.wires.tolist()

##############################################################################
# compute the overlap of the quantum states by first applying the embedding of the first
# datapoint and then the adjoint of the embedding of the second datapoint. We
# finally extract the probabilities of observing each basis state.


@qml.qnode(dev, interface="autograd")
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)


##############################################################################
# The kernel function itself is now obtained by looking at the probability
# of observing the all-zero state at the end of the kernel circuit 


def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]


##############################################################################
# Before focusing on the kernel values we have to provide values for the
# variational parameters. 

init_params = random_params(num_wires=5, num_layers=6)

##############################################################################
# Now we can have a look at the kernel value between the first and the second datapoint:

kernel_value = kernel(X[0], X[1], init_params)
print(f"The kernel value between the first and second datapoint is {kernel_value:.3f}")

##############################################################################

init_kernel = lambda x1, x2: kernel(x1, x2, init_params)
K_init = qml.kernels.square_kernel_matrix(X, init_kernel, assume_normalized_kernel=True)

with np.printoptions(precision=3, suppress=True):
    print(K_init)



##############################################################################
#supply sklearn.svm.SVC with a function that takes two sets of datapoints and returns the associated kernel matrix.
# let scikit-learn adjust the SVM from our Quantum Embedding Kernel.


svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(X, Y)

##############################################################################
# measure which percentage of the dataset it classifies correctly.


def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)


accuracy_init = accuracy(svm, X, Y)
print(f"The accuracy of the kernel with random parameters is {accuracy_init:.3f}")




##############################################################################
# In summary, the kernel-target alignment effectively captures how well
# the kernel you chose reproduces the actual similarities of the data.
# having a high kernel-target alignment is only a necessary but not a sufficient condition for a good
# performance of the kernel. This means having good alignment is
# guaranteed for good performance, but optimal alignment will not always
# bring optimal training accuracy with it.


kta_init = qml.kernels.target_alignment(X, Y, init_kernel, assume_normalized_kernel=True)

print(f"The kernel-target alignment for our dataset and random parameters is {kta_init:.3f}")

##############################################################################
# improve the kernel-target alignment


def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """Kernel-target alignment between kernel and labels."""

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm

    return inner_product


params = init_params
opt = qml.GradientDescentOptimizer(0.2)

for i in range(500):
    # Choose subset of datapoints to compute the KTA on.
    subset = np.random.choice(list(range(len(X_train))), 4)
    # Define the cost function for optimization
    cost = lambda _params: -target_alignment(
        X_train[subset],
        Y_train[subset],
        lambda x1, x2: kernel(x1, x2, _params),
        assume_normalized_kernel=True,
    )
    # Optimization step
    params = opt.step(cost, params)

    # Report the alignment on the full dataset every 50 steps.
    if (i + 1) % 50 == 0:
        current_alignment = target_alignment(
            X,
            Y,
            lambda x1, x2: kernel(x1, x2, params),
            assume_normalized_kernel=True,
        )
        print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

##############################################################################
# assess the impact of training the parameters of the quantum kernel

# First create a kernel with the trained parameter baked into it.
trained_kernel = lambda x1, x2: kernel(x1, x2, params)

# Second create a kernel matrix function using the trained kernel.
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)

# Note that SVC expects the kernel argument to be a kernel matrix function.

svm_train = SVC(kernel=trained_kernel_matrix).fit(X_train, Y_train)
#svm_test = SVC(kernel=trained_kernel_matrix).fit(X_test, Y_test)
pred_array_train = svm_train.predict(X_train)
pred_array_test = svm_train.predict(X_test)

##############################################################################
#accuracy_trained = accuracy(svm_trained, X, Y)
accuracy_train = 1 - np.count_nonzero(pred_array_train - Y_train) / len(Y_train)
accuracy_test = 1 - np.count_nonzero(pred_array_test - Y_test) / len(Y_test)

print(f"The accuracy of a kernel for training sets with trained parameters is {accuracy_train:.3f}")
print(f"The accuracy of a kernel for testing sets with trained parameters is {accuracy_test:.3f}")
np.save("/work/hsukaicheng/carbon fiber/QML/QK_v00/final_weights_v00", params)
np.save("/work/hsukaicheng/carbon fiber/QML/QK_v00/pred_array_train_v00", pred_array_train)
np.save("/work/hsukaicheng/carbon fiber/QML/QK_v00/pred_array_test_v00", pred_array_test)
np.save("/work/hsukaicheng/carbon fiber/QML/QK_v00/train_indices_v00", train_indices)
np.save("/work/hsukaicheng/carbon fiber/QML/QK_v00/test_indices_v00", test_indices)
