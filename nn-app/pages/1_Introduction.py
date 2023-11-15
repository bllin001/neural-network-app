import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

#=======================================================================================================================#

st.set_option('deprecation.showPyplotGlobalUse', False)

#=======================================================================================================================#


st.title("Neural Network Tutorial")

#=======================================================================================================================#
st.header("Introduction")

#=======================================================================================================================#
st.header('Activation Functions')
# What is an activation function?
'''
Activation functions are mathematical equations that determine the output of a neural network.
The function is attached to each neuron in the network, and determines whether it should be activated (“fired”) or not, based on whether each neuron’s input is relevant for the model’s prediction.
Activation functions also help normalize the output of each neuron to a range between 1 and 0 or between -1 and 1.
The activation function is a non-linear transformation that we do over the input signal.
It can be simply understood as a decision making function.
The activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias with it.
The purpose of the activation function is to introduce non-linearity into the output of a neuron.
This is important because most real world data is non linear and we want neurons to learn these non linear representations.
Without an activation function our model would simply be a linear regression model, which has limited power and does not perform good most of the times.
Therefore, we use a non-linear activation function which can map any given input to non-linear space.
Some of the popular activation functions are:
1. Linear
4. Sigmoid
1. Binary Step
2. Piecewise Linear
3. Bipolar
5. Bipolar Sigmoid
6. Tanh
7. ArcTan
8. ReLU
9. Leaky ReLU
10. ELU
11. SoftPlus
13. Softmax

where $z$ is the weighted sum of inputs and bias, define by:

$$z = \mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^{n} = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$$
'''

activation = st.selectbox('Select an activation function', 
                          ['Linear', 'Sigmoid', 'Binary Step', 'Piecewise Linear','Bipolar', 'Bipolar Sigmoid',
                            'Tanh', 'ArcTan', 'ReLU', 'Leaky ReLU', 'ELU', 'SoftPlus', 'Softmax',])

def plot(x, y):
    plt.plot(x, y, c='r', lw=3)
    plt.xlabel('z')
    plt.ylabel('f(z)')
    plt.grid()
    st.pyplot()

if activation == 'Binary Step':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = \begin{cases} 0 & \text{if } z \leq 0 \\ 1 & \text{if } z > 0 \end{cases}''')
    # Derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = \begin{cases} 0 & \text{if } z \neq 0 \\ \text{undefined} & \text{if } z = 0 \end{cases}''')
    # Plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = np.heaviside(x, 1)
    plot(x, y)

elif activation == 'Piecewise Linear':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = \begin{cases} 0 & \text{if } z \leq 0 \\ z & \text{if } 0 < z < 1 \\ 1 & \text{if } z \geq 1 \end{cases}''')
    # Derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = \begin{cases} 0 & \text{if } z \leq 0 \\ 1 & \text{if } 0 < z < 1 \\ 0 & \text{if } z \geq 1 \end{cases}''')
    # Plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = np.piecewise(x, [x <= 0, (x > 0) & (x < 1), x >= 1], [0, lambda x: x, 1])
    plot(x, y)

elif activation == 'Bipolar':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = \begin{cases} -1 & \text{if } z \leq 0 \\ 1 & \text{if } z > 0 \end{cases}''')
    # Derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = \begin{cases} 0 & \text{if } z \neq 0 \\ \text{undefined} & \text{if } z = 0 \end{cases}''')
    # Plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = np.piecewise(x, [x <= 0, x > 0], [-1, 1])
    plot(x, y)

elif activation == 'Sigmoid':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = \frac{1}{1 + e^{-z}}''')
    # derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = \frac{e^{-z}}{(1 + e^{-z})^2}''')
    # plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    plot(x, y)

elif activation == 'Bipolar Sigmoid':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = \frac{1 - e^{-z}}{1 + e^{-z}}''')
    # Derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = \frac{2e^{-z}}{(1 + e^{-z})^2}''')
    # Plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = (1 - np.exp(-x)) / (1 + np.exp(-x))
    plot(x, y)

elif activation == 'Tanh':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}''')
    # Derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = 1 - f(z)^2''')
    # Plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    plot(x, y)

elif activation == 'ArcTan':
    # Activation function
    st.latex(r'''f(z) = tan^{-1}(z)''')
    # Derivate
    st.latex(r'''f'(z) = \frac{1}{1 + z^2}''')
    # Plot
    x = np.linspace(-10, 10, 100)
    y = np.arctan(x)
    plot(x, y)

elif activation == 'ReLU':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = max(0, z)''')
    # Derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = \begin{cases} 0 & \text{if } z < 0 \\ 1 & \text{if } z \geq 0 \end{cases}''')
    # Plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = np.maximum(0, x)
    plot(x, y)

elif activation == 'Leaky ReLU':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = \begin{cases} 0.01z & \text{if } z < 0 \\ z & \text{if } z \geq 0 \end{cases}''')
    # Derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = \begin{cases} 0.01 & \text{if } z < 0 \\ 1 & \text{if } z \geq 0 \end{cases}''')
    # Plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = np.maximum(0.01 * x, x)
    plot(x, y)

elif activation == 'ELU':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = \begin{cases} \alpha(e^z - 1) & \text{if } z < 0 \\ z & \text{if } z \geq 0 \end{cases}''')
    # Derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = \begin{cases} \alpha e^z & \text{if } z < 0 \\ 1 & \text{if } z \geq 0 \end{cases}''')
    # Plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = np.piecewise(x, [x < 0, x >= 0], [lambda x: 0.5 * (np.exp(x) - 1), lambda x: x])
    plot(x, y)

elif activation == 'SoftPlus':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = ln(1 + e^z)''')
    # Derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = \frac{1}{1 + e^{-z}}''')
    # Plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = np.log(1 + np.exp(x))
    plot(x, y) 

elif activation == 'Linear':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = z''')
    # Derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = 1''')
    # Plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = x
    plot(x, y)

elif activation == 'Softmax':
    # Activation function
    st.write('Equation:')
    st.latex(r'''f(z) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}''')
    # Derivate
    st.write('Derivative:')
    st.latex(r'''f'(z) = f(z)(1 - f(z))''')
    # Plot
    st.write('Plot:')
    x = np.linspace(-10, 10, 100)
    y = np.exp(x) / np.sum(np.exp(x))
    plot(x, y)

#=======================================================================================================================#
