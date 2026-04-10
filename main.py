
## Imported Python Libraries: Depending on your version or installation you may have to
## employ a virtual environment done in the cmd terminal to import the libraries used
##  STEPS to install all necessary libraries in VSCode:
##  | 1 run new terminal
##  | 2 click on plus sign
##  | 3 select cmd or command terminal
##  | 4 run 'python -m venv [ENVIRONMENT_NAME]' in command line         - creates virtual environment for workspace
##  | 5 click yes on notification to bottom right to change python version to that of your virtual environment
##  | 6 run '\[ENVIRONMENT_NAME]\Scripts\Activate' in command line      - runs activate script in virtual environment to be
##  |                                                                   - able to reference its contents when running commands
##  | 7 run 'pip install [LIBRARY_NAME]' for each library listed below

import numpy as np                  # import Numpy lets us do vairous calculations and linear algebra operatons
import tensorflow as tf             # import Tensorflow: python library full of functions for training neural networks
import matplotlib.pyplot as plt     # import matplotlib: python library full of functions and attributes for generating graphs
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import scipy.optimize               # import Scipy.optimize: python package full of functions for linear algebra and calculus
                                    #   type optimization, we will apply BFGS to get a gradient for our neural network layer eqns
                                    #   which will help us to find an optimal linear combination of our neuron values
from layer import GradientLayer
from optimize import L_BFGS_B

# ----------------------------------Funcs-----------------------------------------------------------------------------------------
def uInit(tx, c = 1):               # Main way we can interact with our program here
    t = tx[..., 0, None]            # t is the 0th column of tx which we set to 0 before calling
    x = tx[..., 1, None]            # x is the 1th column of tx which we set to be a collection of random numbers on [0,l)
    return tf.sin(3.14159*x/4)/2    # u(x, 0) = (1/2)sin(pi x/4) from ex. 1 on our pde packet

def vInit(tx):                      # Another main way we can interact with our program here
    t = tx[..., 0, None]            # t is the 0th column of tx which we set to 0 before calling
    x = tx[..., 1, None]            # x is the 1th column of tx which we set to be a collection of random numbers on [0,l)
    return - tf.sin(3.14159*x/2)

# 2 function below are more back-end procedures
def buildNetwork(num_inputs=2, layers=[32, 16, 16, 32], num_outputs=1): # returns a network of layers I guess lol
    # input layer
    inputs = tf.keras.layers.Input(shape=(num_inputs,))
    # hidden layers
    x = inputs
    for layer in layers:
        x = tf.keras.layers.Dense(layer, activation='tanh',
            kernel_initializer='he_normal')(x)
    # output layer
    outputs = tf.keras.layers.Dense(num_outputs,
        kernel_initializer='he_normal')(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

def buildPinn(network, grad, c):
    # equation input: (t, x)
    tx_eqn = tf.keras.layers.Input(shape=(2,))
    # initial condition input: (t=0, x)
    tx_ini = tf.keras.layers.Input(shape=(2,))
    # boundary condition input: (t, x=-1) or (t, x=+1)
    tx_bnd = tf.keras.layers.Input(shape=(2,))

    # compute gradients
    _, _, _, d2u_dt2, d2u_dx2 = grad(tx_eqn)

    # equation output being zero
    u_eqn = d2u_dt2 - c*c * d2u_dx2
    # initial condition output
    u_ini, du_dt_ini, _, _, _ = grad(tx_ini)
    # boundary condition output
    u_bnd = network(tx_bnd)  # dirichlet
    # _, _, u_bnd, _, _ = grad(tx_bnd)  # neumann

    # build the PINN model for the wave equation
    return tf.keras.models.Model(
        inputs=[tx_eqn, tx_ini, tx_bnd],
        outputs=[u_eqn, u_ini, du_dt_ini, u_bnd])
#---------------------------------------------------------------------------------------------------------------------------------

#----------------------------------Script-----------------------------------------------------------------------------------------
print("hello")
#--------Variables----------#
num_train_samples = 10000
num_test_samples = 1000
c = 1                   # wave speed
l = 4                   # length of string
x_o = 0                 # x coordinate of the left bound of the string
time_length = 1.333     # length of time
#---------------------------#

# build a core network model
network = buildNetwork()
network.summary()
# build a PINN model
grads= GradientLayer(network)
pinn = buildPinn(network, grads, 1)

# create training input
tx_eqn = np.random.rand(num_train_samples, 2)   # This function just makes a matrix with 2 columns and num_train_samples 
                                                # rows with a random given value between 0 and 1
tx_eqn[..., 0] = time_length*tx_eqn[..., 0]
tx_eqn[..., 1] = l*tx_eqn[..., 1] - x_o
tx_ini = np.random.rand(num_train_samples, 2)
tx_ini[..., 0] = 0                              # t_o = 0
tx_ini[..., 1] = l*tx_ini[..., 1] - x_o
tx_bnd = np.random.rand(num_train_samples, 2)
tx_bnd[..., 0] = 4*tx_bnd[..., 0]
tx_bnd[..., 1] = l*np.round(tx_bnd[..., 1]) - x_o

# create training output
u_zero = np.zeros((num_train_samples, 1))
u_ini = uInit(tf.constant(tx_ini)).numpy()
du_dt_ini = vInit(tf.constant(tx_ini)).numpy()

x_train = [tx_eqn, tx_ini, tx_bnd]
y_train = [u_zero, u_ini, du_dt_ini, u_zero]
opper = L_BFGS_B(pinn, x_train, y_train)
opper.fit()

# predict u(t,x) distribution
t_flat = np.linspace(0, time_length, num_test_samples)
x_flat = np.linspace(x_o, l, num_test_samples)
t, x = np.meshgrid(t_flat, x_flat)
tx = np.stack([t.flatten(), x.flatten()], axis=-1)
u = network.predict(tx, batch_size=num_test_samples)
u = u.reshape(t.shape)

# plot u(t,x) distribution as a color-map
fig = plt.figure(figsize=(7,4))
gs = GridSpec(2, 3)
plt.subplot(gs[0, :])
vmin, vmax = -0.5, +0.5
plt.pcolormesh(t, x, u, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label('u(t,x)')
cbar.mappable.set_clim(vmin, vmax)
# plot u(t=const, x) cross-sections
cross_time = time_length/3
t_cross_sections = [cross_time, 2*cross_time, 3*cross_time]
for i, t_cs in enumerate(t_cross_sections):
    plt.subplot(gs[1, i])
    tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    plt.plot(x_flat, u)
    plt.title('t={}'.format(t_cs))
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
plt.tight_layout()
plt.savefig('result_img_dirichlet.png', transparent=True)
plt.show()