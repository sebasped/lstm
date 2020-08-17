#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# separa datos temporales para aprendizaje supervisado
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
  
    
# definición del modelo
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=20):
        super().__init__()
        self.output_size = input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, self.output_size)
            
    def forward(self,x):
        x, (h,c) = self.lstm(x)
        x = self.fc(x)
        return x, self.fc(h)



if __name__ == '__main__':
  
    # matriz de datos artificiales. Cada fila es un tiempo.
    dataOrig = np.linspace(start=(0,5),stop=(100,105), num=15,dtype=int)
    
    # con cuántas filas se predice la siguiente
    n_steps = 3
    
    # se separan los datos para aprendizaje supervisado.
    X, y = split_sequence(dataOrig, n_steps)
    
    batch_size = X.shape[0]
    seq_len = X.shape[1]
    input_size = X.shape[2]
    
    # para a tensor de torch
    x_trn = torch.FloatTensor(X).view(batch_size,seq_len,input_size)
    labels_trn = torch.FloatTensor(y)
    
    B=3 #tamaño del batch
    trn_data = TensorDataset(x_trn, labels_trn)
    trn_load = DataLoader(trn_data, shuffle=True, batch_size=B)
    
    model = LSTM(input_size)
    costF = torch.nn.MSELoss() 
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    T = 1000
    model.train()
    for t in range(T+1):
        for data, label in trn_load:
            # reinicializo el gradiente
            optim.zero_grad()
            
            # calculo predicción por modelo
            outx, outh = model(data)
            outh = outh.squeeze()
            
            # comparo contra target verdadero
            error = costF(outh, label)
            # gradiente por back prop
            error.backward()
            # paso optimización
            optim.step()
            
        if t%100==0 or t==T:
            print(t)
            print(error.item())
            print(label)
            print(outh)
            print('*******')


    
    # predicción
    model.eval()
    with torch.no_grad():
        # newInput = torch.tensor([5,10,15], dtype=torch.float32).view(1,3,1)
        newInput = torch.FloatTensor([[110,115],[120,125],[130,135]]).view(1,seq_len,input_size)
        outx, outh = model(newInput)
        print(outh)