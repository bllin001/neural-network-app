import streamlit as st
import numpy as np
import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt
import time
from Perceptron import Perceptron
from Data import data

#=======================================================================================================================#

st.set_option('deprecation.showPyplotGlobalUse', False)

#=======================================================================================================================#

#------------------ Sidebar ------------------#
st.sidebar.title("Simulation Parameters")

# Add a selectbox to the sidebar:
case = st.sidebar.radio(
    'Select the logic gate you want to simulate',
    ('AND', 'OR','XOR')
)

df = data(case)

# add to x a column of 1s for bias term
X = np.c_[np.ones((df.X.shape[0])), df.X]
y = df.y

# Add a selectbox to the sidebar:
activation = st.sidebar.selectbox(
    'Select an activation function',
    ('binary_step', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'softmax')
)

# Add a textbox to the sidebar:
epoch = st.sidebar.text_input(
    'Enter number of epochs',
    '5'
)

epoch = int(epoch)


# Add a slider to the sidebar:
learning_rate = st.sidebar.slider(
    'Select a learning rate $\\alpha$',
    0.0, 1.0, (0.25)
)

learning_rate = float(learning_rate)

# Add three inputs text widget to the sidebar:
wo = st.sidebar.text_input('Enter initial bias ($w_0$)', '0')
w1 = st.sidebar.text_input('Enter initial weight 1 ($w_1$)', '0')
w2 = st.sidebar.text_input('Enter initial weight 2 ($w_2$)', '0')

w = [float(wo), float(w1), float(w2)]

# Add start button to the sidebar:
# start = st.sidebar.button('Run')

#=======================================================================================================================#

#------------------ Main Page ------------------#

st.title("Single Perceptron using " + activation + " Activation Function")

'''
In this section, you can simulate a single perceptron for a simple classification task.

You can use logic gates as a dataset. The perceptron will learn to classify the inputs into their respective logic gates.
'''

#------------------ Neural Network Architecture ------------------#

st.header('Neural Network Architecture')

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

## add edges to the graph
dot.edge('x1', 'y', label='w1') 
dot.edge('x2', 'y', label='w2')
dot.edge('1', 'y', label='w0')

## render the graph
dot

#------------------ Simulation ------------------#

st.header('Simulation case: ' + case + ' gate')

#------------------ Dataset ------------------#

def render_data_frame(X, y):
    df = pd.DataFrame(X, columns=['x1','x2'])
    df['y'] = y
    return df

if st.checkbox('Show dataframe'):
    st.dataframe(render_data_frame(df.X, df.y))

#------------------ Run the simulation ------------------#

run = st.toggle('Run Simulation')


if run:
    # Train the model
    model = Perceptron(epochs=epoch, learning_rate=learning_rate)
    model.fit(X, y, w, activation=activation)

    st.subheader('Simulation Results')

    #------------------ Show weights ------------------#

    # Show the weights as text (i.e Bias is: w0, W1 is: w1, W2 is: w2)
    col1, col2, col3 = st.columns([2,2,2])

    with col1:
        st.write('Final bias is: $w_0$ = ', model.w[0])

    with col2:
        st.write('Final weight 1 is: $w_1$ = ', model.w[1])

    with col3:
        st.write('Final weigt is: $w_2$ = ', model.w[2])

    #------------------ Show Neural Network Architecture ------------------#    
    
    nn_plot = st.checkbox('Show Neural Network Architecture')

    if nn_plot:
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

        # add edges to the graph with weights
        dot.edge('1', 'y', label=f'w0 = {model.w[0]:.3f}')
        dot.edge('x1', 'y', label=f'w1 = {model.w[1]:.3f}')
        dot.edge('x2', 'y', label=f'w2 = {model.w[2]:.3f}')

        # render the graph
        dot

    st.subheader('Simulation Iterations')
    
    #------------------ Show iterations ------------------#

    nn_iterations = st.checkbox('Show Perceptron Iterations')

    if nn_iterations:
        
        model.iteration['y_hat'] = model.iteration['y_hat'].astype('int64')

        if epoch == 1:
            st.write(model.iteration[model.iteration['Epoch'] == 1].drop(columns=['Epoch']))
        else:
            # Slider between 1 and epoch, step 1
            nn_epochs = st.slider('Select the epoch you want to see', 1, epoch, 1)

            # Remove index column
            iteration_df = model.iteration[model.iteration['Epoch'] == nn_epochs].reset_index(drop=True)
            # Don't show the Epoch column
            iteration_df = iteration_df.drop(columns=['Epoch'])
            
            st.write(iteration_df)

    #------------------ Show loss ------------------#

    nn_loss = st.checkbox('Show Training Loss')

    if nn_loss:
        # Plot the loss
        plt.plot(model.loss['Epoch'], model.loss['Total_Loss'], c='r', lw=3)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Training Loss')
        plt.grid()
        # Set axis as integer instead of float
        plt.locator_params(axis='x', integer=True)
        plt.locator_params(axis='y', integer=True)
        st.pyplot()

    #------------------ Show prediction ------------------#

    st.subheader('Test the model')
    prediction = st.checkbox('Show Prediction')

    if prediction:
        prediction_data = []
        for i in range(4):
            prediction_data.append([X[i][1], case, X[i][2], y[i], model.predict(X)[i]])

        prediction_df = pd.DataFrame(prediction_data, columns=['X1', 'Logic Gate', 'X2', 'y', 'Prediction'])

        styled_df = prediction_df.style
        st.dataframe(styled_df)




