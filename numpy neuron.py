import numpy as np
inputs = [ 1.0 , 2.0 , 3.0 , 2.5 ]
weights = [ 0.2 , 0.8 , - 0.5 , 1.0 ]
bias = 2.0
outputs = np.dot(weights, inputs) + bias
print (outputs)


import numpy as np
inputs = [ 1.0 , 2.0 , 3.0 , 2.5 ]
weights = [[ 0.2 , 0.8 , - 0.5 , 1 ],
[ 0.5 , - 0.91 , 0.26 , - 0.5 ],
[ - 0.26 , - 0.27 , 0.17 , 0.87 ]]
biases = [ 2.0 , 3.0 , 0.5 ]
layer_outputs = np.dot(weights, inputs) + biases
print (layer_outputs)


import numpy as np
inputs =    [[ 1.0 , 2.0 , 3.0 , 2.5 ],
            [ 2.0 , 5.0 , - 1.0 , 2.0 ],
            [ - 1.5 , 2.7 , 3.3 , - 0.8 ]]
weights =   [[ 0.2 , 0.8 , - 0.5 , 1.0 ],
            [ 0.5 , - 0.91 , 0.26 , - 0.5 ],
            [ - 0.26 , - 0.27 , 0.17 , 0.87 ]]
biases = [ 2.0 , 3.0 , 0.5 ]
layer_outputs = np.dot(inputs, np.array(weights).T) + biases
print (layer_outputs)


import numpy as np
inputs = [[ 1 , 2 , 3 , 2.5 ], [ 2. , 5. , - 1. , 2 ], [ - 1.5 , 2.7 , 3.3 , - 0.8 ]]
weights =   [[ 0.2 , 0.8 , - 0.5 , 1 ],
            [ 0.5 , - 0.91 , 0.26 , - 0.5 ],
            [ - 0.26 , - 0.27 , 0.17 , 0.87 ]]
biases = [ 2 , 3 , 0.5 ]
weights2 =  [[ 0.1 , - 0.14 , 0.5 ],
            [ - 0.5 , 0.12 , - 0.33 ],
            [ - 0.44 , 0.73 , - 0.13 ]]
biases2 = [ - 1 , 2 , - 0.5 ]
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print (layer2_outputs)


import numpy as np
import nnfs
nnfs.init()
print (np.random.randn( 200, 1 ))


import numpy as np
import nnfs
nnfs.init()
n_inputs = 2
n_neurons = 4
weights = 0.01 * np.random.randn(n_inputs, n_neurons)
biases = np.zeros(( 1 , n_neurons))
print (weights)
print (biases)


import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
# Dense layer
class Layer_Dense :
    # Layer initialization
    def __init__ ( self , n_inputs , n_neurons ):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(( 1 , n_neurons))
    # Forward pass
    def forward ( self , inputs ):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
# Create dataset
X, y = spiral_data( samples = 100 , classes = 3 )
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense( 2 , 3 )
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Let's see output of the first few samples:
print (dense1.output[: 50 ])
