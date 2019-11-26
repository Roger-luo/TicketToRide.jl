import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import torch

N = int(input('N = '))
nlayers = int(input('# of layers = '))

dev = qml.device('default.qubit', wires=N)
nparams = 3*N*nlayers 

def starting_angles():
    # for each layer
    # 3*N params per layer
    params = np.zeros((nlayers, 3*N))
    for L in range(nlayers):
        params[L] = np.random.uniform(low=0, high=2*np.pi, size=3*N)

    return params

def layer(params, layer=None):
    for i in range(N):
        qml.RZ(params[layer][i], wires = i)
        qml.RX(params[layer][N + i], wires = i)
        qml.RZ(params[layer][2*N + i], wires = i)
        
        # PBC's
        if i == N-1:
            qml.CNOT(wires = [i,0])
        else:
            qml.CNOT(wires = [i,i+1])

def ansatz(params, nlayers=100):
    for L in range(nlayers):
        layer(params, layer=L)

@qml.qnode(dev)
def circuit(params, i=None, pauli=None):
    ansatz(params, nlayers=nlayers)
    return qml.expval( pauli(i) @ pauli((i+1) % N) )

def cost(params):

    result = sum(circuit(params, i=i, pauli=qml.PauliZ) for i in range(N))
    result += sum(circuit(params, i=i, pauli=qml.PauliY) for i in range(N))
    result += sum(circuit(params, i=i, pauli=qml.PauliX) for i in range(N))

    return result 

#opt = qml.QNGOptimizer(stepsize = 0.01)
opt = AdamOptimizer(stepsize=0.1)
steps = 100
params = starting_angles()

for s in range(steps):
    
    #print(opt.step(cost, params))
    params = opt.step(cost, params)

    print('Step\tEnergy / N')
    print('{}\t{}'.format(s, cost(params)/(8*N)))
