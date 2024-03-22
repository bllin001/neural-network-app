import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

#=======================================================================================================================#

st.set_option('deprecation.showPyplotGlobalUse', False)

#=======================================================================================================================#


st.title("Neural Network Tutorial")

#=======================================================================================================================#
st.header("Architecture")

architecture = st.selectbox('Select an architecture', ['Perceptron', 'Multi-Layer Perceptron'])

if architecture == 'Perceptron':
    st.write('This is a perceptron with 2 inputs and 1 output:')
    # create a new Digraph object
    dot = Digraph(comment='Neural Network')
    dot.attr(rankdir='LR')
    dot.attr(splines='line')
    dot.attr('node', shape='circle', fixedsize='true', width='0.6', height='0.6')

    with dot.subgraph(name='cluster_0') as c:
        c.attr(color='white', label='Input Layer')
        c.attr('node', shape='triangle')
        c.node('1', '1', style='filled', fillcolor='blue')
        c.attr('node', shape='circle')
        c.node('x1', 'x1', style='filled', fillcolor='red')
        c.node('x2', 'x2', style='filled', fillcolor='red')
        

    with dot.subgraph(name='cluster_2') as c:
        c.attr(color='white', label='Output Layer')
        c.node('y', 'y', style='filled', fillcolor='green')

    # add edges to the graph
    dot.edge('x1', 'y', label='w1') 
    dot.edge('x2', 'y', label='w2')
    dot.edge('1', 'y', label='w0')

    # render the graph
    dot

else:
    st.write('This is a multi-layer perceptron with 2 inputs, 2 hidden units and 1 output:')
    # create a new Digraph object
    dot = Digraph(comment='Neural Network')
    dot.attr(rankdir='LR')
    dot.attr(splines='line')
    dot.attr('node', shape='circle', fixedsize='true', width='0.6', height='0.6')

    with dot.subgraph(name='cluster_0') as c:
        c.attr(color='white', label='Input Layer')
        c.node('x1', 'x1', style='filled', fillcolor='red')
        c.node('x2', 'x2', style='filled', fillcolor='red')

    with dot.subgraph(name='cluster_1') as c:
        c.attr(color='white', label='Hidden Layer')
        c.node('h1', 'h1', style='filled', fillcolor='blue')
        c.node('h2', 'h2', style='filled', fillcolor='blue')

    with dot.subgraph(name='cluster_2') as c:
        c.attr(color='white', label='Output Layer')
        c.node('y', 'y', style='filled', fillcolor='green')

    # add edges to the graph
    dot.edge('x1', 'h1', label='w1')
    dot.edge('x1', 'h2', label='w2')
    dot.edge('x2', 'h1', label='w3')
    dot.edge('x2', 'h2', label='w4')
    dot.edge('h1', 'y', label='w5') 
    dot.edge('h2', 'y', label='w6')

    # render the graph
    dot

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
'''

col1, col2 = st.columns([2,2])

with col1:
    st.write('''
            1. Linear
            2. Sigmoid
            3. Binary Step
            4. Piecewise Linear
            5. Bipolar
            6. Bipolar Sigmoid
            7. Tanh
            ''' )

with col2:
    st.write('''
        7. ArcTan
        8. ReLU
        9. Leaky ReLU
        10. ELU
        11. SoftPlus
        13. Softmax
        ''')

'''
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
st.header('Steps to train a neural network')

st.subheader('1. Initialize the model')

'''
We are going to initialize the model with the following parameters:
- $w_0$
- $w_1$
- $w_2$
- $\\alpha$
- $Epochs$
'''

st.subheader('2. Forward Propagation')

'''

Node $y$ is define by:
- Input:
$$z=w_1X_1+w_2X_2+b$$
- Output: 
$$\hat{y}=f_1(z_1)$$

where $f_1$ is the activation function, in this case is the **Binary Step** function.

'''

st.subheader('3. Calculate the error')

'''

Total error is define by: 

$$E_{total}=\\frac{1}{2}\sum_{i=1}^{n}(\hat{y_i}-y_i)^2$$

'''

st.subheader('4. Backward Propagation')

'''

We need to update the weights of the network, and this can be expressed as:
$$w_i^*=w_i+\Delta w_i$$
The $\Delta w_i$ is estimated as: 
$$\Delta w_i=\\alpha(y_i-\hat{y_i})*x_i$$
where: 
- $w_i$ is the weight for case $i$
- $\\alpha$ is the learning rate, when $\\alpha \in \{0,1\}$
- $y$ is the actual value ("true class")
- $\hat{y}$ is the predicted value ("predicted class")
- $x_i$ is the vector of inputs for case $i$

We have three condition in order to update the weights:
1. **No updated $w_1$:** When $\hat{y_i}=y_i$, it is not necessary updated $w_i$, because: $(y_i-\hat{y_i})=(y_i-y_i)=0$, then $\Delta w_i=0$, at the end: $$w_i^*=w_i$$
2. **Decrease $w_i$:** When $\hat{y_i}>y_i$, means that $y_i=0$ and $\hat{y}=1$, then $(y_i-\hat{y_i})=(0-1)=-1$, then $\Delta w_i=-\\alpha x_1$, at the end $$w_i^*=w_i-\\alpha x_1$$
3. **Increase $w_i$:** When $\hat{y_i}<y_i$, means that $y_i=1$ and $\hat{y}=0$, then $(y_i-\hat{y_i})=(1-0)=1$, then $\Delta w_i=\\alpha x_1$, at the end $$w_i^*=w_i+\\alpha x_1$$
'''
#=======================================================================================================================#

st.header('Example - Logic gates: XOR')

'''
We are going to explore the XOR case with the following data:

| $x_1$ | $x_2$ | y |
|---|---|---------|
| 0 | 0 |    0    |
| 1 | 0 |    1    |
| 0 | 1 |    1    |
| 1 | 1 |    0    |
'''

st.subheader('1. Initialize the model')

'''
- $w_0=0$
- $w_1=0$
- $w_2=0$
- $\\alpha=1$
- $Epochs=1$
'''

st.subheader('2. Forward Propagation')
st.write('**First row**: $x_1=0$ and $x_2=0$, then:')

'''
#### **First row**: $x_1=0$ and $x_2=0$, then:


- Input:
$$z = w1 * x1 + w2 * x2 + w0$$

$$z = 0 * 0 + 0 * 0 + 0$$

$$z = 0$$

- Output:
$$\hat{y} = f(z)$$

$$\hat{y} = f(0)$$

$$\hat{y} = 0$$

##### Total error (Cost Function):
$$E_{total} = \\frac{1}{2}(\hat{y} - y)^2$$

'''


9