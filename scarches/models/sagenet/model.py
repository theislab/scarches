import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn



class NN(nn.Module):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[], \
        n_hidden_FC=[10], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(NN, self).__init__()
        self.FC           = True
        self.n_features   = n_features
        self.n_classes    = n_classes
        self.layers_GNN   = nn.ModuleList()
        self.layers_FC    = nn.ModuleList()
        self.n_layers_GNN = len(n_hidden_GNN)
        self.n_layers_FC  = len(n_hidden_FC)
        self.dropout_GNN  = dropout_GNN
        self.dropout_FC   = dropout_FC
        self.n_hidden_GNN = n_hidden_GNN
        self.n_hidden_FC  = n_hidden_FC
        self.conv         = False
        if self.n_layers_GNN > 0:
            self.FC = False

        # Fully connected layers. They occur after the graph convolutions (or at the start if there no are graph convolutions)
        if self.n_layers_FC > 0:
            if self.n_layers_GNN==0:
                self.layers_FC.append(nn.Linear(n_features, n_hidden_FC[0]))
            else:
                self.layers_FC.append(nn.Linear(n_features*n_hidden_GNN[-1], n_hidden_FC[0]))
            if self.n_layers_FC > 1:
                for i in range(self.n_layers_FC-1):
                    self.layers_FC.append(nn.Linear(n_hidden_FC[i], n_hidden_FC[(i+1)]))

        # Last layer
        if self.n_layers_FC>0:
            self.last_layer_FC = nn.Linear(n_hidden_FC[-1], n_classes)
        elif self.n_layers_GNN>0:
            self.last_layer_FC = nn.Linear(n_features*n_hidden_GNN[-1], n_classes)
        else:
            self.last_layer_FC = nn.Linear(n_features, n_classes)

    def forward(self,x,edge_index):
        if self.FC:
            # Resize from (1,batch_size * n_features) to (batch_size, n_features)
            x = x.view(-1,self.n_features)
        if self.conv:
            x = x.view(-1,1,self.n_features)
            for layer in self.layers_GNN:
                x = F.relu(layer(x))
                x = F.max_pool1d(x, kernel_size=self.pool_K, stride=1, padding=self.pool_K//2, dilation=1)
                x = F.dropout(x, p=self.dropout_GNN, training=self.training)
            # x = F.max_pool1d(x)
        else:
            for layer in self.layers_GNN:
                x = F.relu(layer(x, edge_index))
                x = F.dropout(x, p=self.dropout_GNN, training=self.training)
        if self.n_layers_GNN > 0:
            x = x.view(-1, self.n_features*self.n_hidden_GNN[-1])
        for layer in self.layers_FC:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout_FC, training=self.training)
        x = self.last_layer_FC(x)
        return x


class GraphSAGE(NN):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[10], \
        n_hidden_FC=[], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(GraphSAGE, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.SAGEConv(1, n_hidden_GNN[0]))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.SAGEConv(n_hidden_GNN[i], n_hidden_GNN[(i+1)]))


class ChebNet(NN):
    def __init__(self,
        n_features,
        n_classes,
        n_hidden_GNN=[10],
        n_hidden_FC=[],
        K=4,
        dropout_GNN=0,
        dropout_FC=0):
        super(ChebNet, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.ChebConv(1, n_hidden_GNN[0], K))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.ChebConv(n_hidden_GNN[i], n_hidden_GNN[(i+1)], K))


class NNConvNet(NN):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[10], \
        n_hidden_FC=[], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(NNConvNet, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.NNConv(1, n_hidden_GNN[0]))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.NNConv(n_hidden_GNN[i], n_hidden_GNN[(i+1)]))

class GATConvNet(NN):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[10], \
        n_hidden_FC=[], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(GATConvNet, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.GATConv(1, n_hidden_GNN[0]))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.GATConv(n_hidden_GNN[i], n_hidden_GNN[(i+1)]))

class GENConvNet(NN):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[10], \
        n_hidden_FC=[], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(GENConvNet, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.GENConv(1, n_hidden_GNN[0]))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.GENConv(n_hidden_GNN[i], n_hidden_GNN[(i+1)]))


class GINConv(NN):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[10], \
        n_hidden_FC=[], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(GINConv, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.GINConv(nn.Sequential(nn.Linear(1, n_hidden_GNN[0]),
                                  nn.ReLU(), nn.Linear(n_hidden_GNN[0],n_hidden_GNN[0])),eps=0.2))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.GINConv(nn.Sequential(nn.Linear(n_hidden_GNN[i], n_hidden_GNN[(i+1)]),
                                  nn.ReLU(), nn.Linear(n_hidden_GNN[(i+1)],n_hidden_GNN[(i+1)]))))

class GraphConv(NN):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[10], \
        n_hidden_FC=[], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(GraphConv, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.GraphConv(1, n_hidden_GNN[0]))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.GraphConv(n_hidden_GNN[i], n_hidden_GNN[(i+1)]))

class MFConv(NN):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[10], \
        n_hidden_FC=[], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(MFConv, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.MFConv(1, n_hidden_GNN[0]))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.MFConv(n_hidden_GNN[i], n_hidden_GNN[(i+1)]))

class TransformerConv(NN):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[10], \
        n_hidden_FC=[], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(TransformerConv, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.TransformerConv(1, n_hidden_GNN[0]))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.TransformerConv(n_hidden_GNN[i], n_hidden_GNN[(i+1)]))

class ConvNet(NN):
    def __init__(self,
        n_features,
        n_classes,
        n_hidden_GNN=[10],
        n_hidden_FC=[],
        filter_K=4,
        pool_K=0,
        dropout_GNN=0,
        dropout_FC=0):
        super(ConvNet, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)
        self.conv = True
        self.filter_K = filter_K
        self.pool_K   = pool_K

        self.layers_GNN.append(nn.Conv1d(in_channels=1, out_channels=n_hidden_GNN[0], kernel_size=filter_K, padding=filter_K//2, dilation=1, stride=1))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(nn.Conv1d(in_channels=n_hidden_GNN[i], out_channels=n_hidden_GNN[(i+1)], kernel_size=filter_K, padding=filter_K//2, dilation=1, stride=1))
