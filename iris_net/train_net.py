from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Hyperparameters for model construction
@dataclass
class Hyperparameters:
    epochs: int
    learning_rate: float
    hidden_layers: any
    activation_function: any
    loss_function: any

# hidden_layers: list of nn.Linear layers 
# activation_function: activation function for the network
class Model(nn.Module):
    def __init__(self,hidden_layers,activation_function):
        super(Model, self).__init__()
        self.input_layer = nn.Linear(4, hidden_layers[0].in_features)
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(hidden_layers[-1].out_features, 3)
        self.activation_function = activation_function
        
    def forward(self, x):
        x = self.activation_function(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))
        x = F.softmax(self.output_layer(x), dim=1)
        return x

# Dump model in onnx format
def dump_onnx(model,filename):
    dummy_input = torch.randn(1,4)
    input_names = [ "iris_inputs" ]
    output_names = [ "iris_class" ]
    torch.onnx.export(model, 
                    dummy_input,
                    filename,
                    verbose=False,
                    input_names=input_names,
                    output_names=output_names,
                    export_params=True,
                    )

# Constructs model from Hyperparameters and runs train test loop
# Returns a trained neural net
def train_test_model(hyper : Hyperparameters ,X_train,y_train,X_test,y_test, plot=False) -> Model:
    # Define model optimizer and loss function
    model     = Model(hyper.hidden_layers,hyper.activation_function)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper.learning_rate)
    loss_fn   = hyper.loss_function

    loss_list     = np.zeros((hyper.epochs,))
    accuracy_list = np.zeros((hyper.epochs,))

    for epoch in trange(hyper.epochs):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss_list[epoch] = loss.item()
        
        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()

    if plot:
        _ , (ax1,ax2) = plt.subplots(1,2)
        ax1.plot(range(len(accuracy_list)),accuracy_list)
        ax2.plot(range(len(loss_list)), loss_list )
        plt.show()
    print(f"Precision: {accuracy_list[-1]}")
    print(f"Loss: {loss_list[-1]}")
    return model

# Load dataset
iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test  = Variable(torch.from_numpy(X_test)).float()
y_test  = Variable(torch.from_numpy(y_test)).long()

# Train models with 1-10 hidden layers all of dimension 50x50
for num_hidden_layers in range(1,101):
    hyper = Hyperparameters(50,0.01,[nn.Linear(50,50)]*num_hidden_layers,F.relu,nn.CrossEntropyLoss())
    model = train_test_model(hyper,X_train,y_train,X_test,y_test)
    model.eval()
    with torch.no_grad():
        dump_onnx(model,f"iris_{num_hidden_layers}_relu_cross-entropy.onnx")