import torch
import matplotlib.pyplot as plt

#Leitura do dataset
#-----------------------------------------------------------------
import pandas as pd
diabetes = pd.read_csv('nn_goaline.csv')
diabetes.head()


#Pre-processamento:
#--------------------------------------------------------------------
from sklearn import preprocessing



diabetes_features=diabetes.loc[:,diabetes.columns!='pl_u']

diabetes_target=diabetes.loc[:,diabetes.columns=='pl_u'].values

# Converte valores para a mesma escala (entre 0 e 1)

print(diabetes_features)

minmax_scale = preprocessing.MinMaxScaler().fit(diabetes_features)
diabetes_features=minmax_scale.transform(diabetes_features)

print(diabetes_features)

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(diabetes_features,diabetes_target)

X_train_tensor=torch.from_numpy(X_train).float()
Y_train_tensor=torch.from_numpy(Y_train).float()

X_test_tensor=torch.from_numpy(X_test).float()
Y_test_tensor=torch.from_numpy(Y_test).float()

#Define a Rede
#----------------------------------------------------------------

from torch.nn import Linear,Tanh,Sequential
D_in,D_hidden,D_out=22,11,1
model = Sequential( Linear(D_in,D_hidden),Linear(D_hidden,D_out) )

#model = Sequential( Linear(D_in,D_hidden),Tanh(),Linear(D_hidden,D_out),Tanh() )
    
minibatch_size=1000

#Inicia os parametros
with torch.no_grad():
    for param in model.parameters(): param.fill_(0.0)
        
import torch.optim as optim

optimizer=optim.ASGD(model.parameters())


#Loop de Treino
#----------------------------------------------------------------   
losses={}
total_loss=0
total_lucro=0
for i in range(0,1000):
    # Seleciona um minibatch aleatório a partir do treino
    minibatch_index=torch.randperm(X_train_tensor.size(0))
    minibatch_index=minibatch_index[:minibatch_size]
    minibatch_x=X_train_tensor[minibatch_index]
    minibatch_y=Y_train_tensor[minibatch_index]

    #Forward pass
    output=model(minibatch_x)
    #loss=F.soft_margin_loss(output,minibatch_y)
    loss=torch.exp(-(torch.log(1+output*minibatch_y.clamp(min=0,max=0.2))).sum())
    # Backward pass
    loss.backward()
    total_loss+=loss.item()
    total_lucro+=(torch.log(1+output*minibatch_y.clamp(min=0,max=0.2))).sum().item()
    # Optimizer pass
    optimizer.step()
    optimizer.zero_grad()
    
    if i>0 and i%50==0:
        losses[i]=total_loss/i
        
        #loss, e o lucro em unidades
        print(losses[i], total_lucro/i )
        

minibatch_x=X_test_tensor[1000:2000]
minibatch_y=Y_test_tensor[1000:2000]
output=model(minibatch_x)
print( (torch.log(1+output*minibatch_y.clamp(min=0,max=0.2))).sum().item()   ) 
