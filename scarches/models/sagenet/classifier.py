import warnings
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from .utils import compute_metrics
from .model import *
from captum import *
from captum.attr import IntegratedGradients
import numpy as np
from sklearn.preprocessing import normalize

class Classifier():
    
    """
        A Neural Network Classifier. A number of Graph Neural Networks (GNN) and an MLP are implemented.
        
        Parameters
        ----------
        n_features : int
            number of input features.
        n_classes : int
            number of classes.
        n_hidden_GNN : list, default=[]
            list of integers indicating sizes of GNN hidden layers.
        n_hidden_FC : list, default=[]
            list of integers indicating sizes of FC hidden layers. If a GNN is used, this indicates FC hidden layers after the GNN layers.
        K : integer, default=4
            Convolution layer filter size. Used only when `classifier == 'Chebnet'`.
        dropout_GNN : float, default=0
            dropout rate for GNN hidden layers.
        dropout_FC : float, default=0
            dropout rate for FC hidden layers.
        classifier : str, default='MLP'
            - 'MLP' --> multilayer perceptron
            - 'GraphSAGE'--> GraphSAGE Network 
            - 'Chebnet'--> Chebyshev spectral Graph Convolutional Network
            - 'GATConv'--> Graph Attentional Neural Network
            - 'GENConv'--> GENeralized Graph Convolution Network
            - 'GINConv'--> Graph Isoform Network
            - 'GraphConv'--> Graph Convolutional Neural Network
            - 'MFConv'--> Convolutional Networks on Graphs for Learning Molecular Fingerprints
            - 'TransformerConv'--> Graph Transformer Neural Network
        lr : float, default=0.001
            base learning rate for the SGD optimization algorithm.
        momentum : float, default=0.9
            base momentum for the SGD optimization algorithm.
        log_dir : str, default=None
            path to the log directory. Specifically, used for tensorboard logs.
        device : str, default='cpu'
            the processing unit.



        See also
        --------
        Classifier.fit : fits the classifier to data
        Classifier.eval : evaluates the classifier predictions
    """
    def __init__(self,
        n_features,
        n_classes,
        n_hidden_GNN=[],
        n_hidden_FC=[],
        K=4,
        pool_K=4,
        dropout_GNN=0,
        dropout_FC=0, 
        classifier='MLP', 
        lr=.001, 
        momentum=.9,
        log_dir=None,
        device='cpu'):
        if classifier == 'MLP': 
            self.net = NN(n_features=n_features, n_classes=n_classes,\
                n_hidden_FC=n_hidden_FC, dropout_FC=dropout_FC)
        if classifier == 'GraphSAGE':
            self.net = GraphSAGE(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN)
        if classifier == 'Chebnet':
            self.net = ChebNet(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN, K=K)
        if classifier == 'GATConv':
            self.net = GATConvNet(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN)
        if classifier == 'GENConv':
            self.net = GENConvNet(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN) 
        if classifier =="GINConv":
            self.net = GINConv(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN) 
        if classifier =="GraphConv":
            self.net = GraphConv(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN) 
        if classifier =="MFConv":
            self.net = MFConv(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN) 
        if classifier =="TransformerConv":
            self.net = TransformerConv(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN)
        if classifier =="Conv1d":
            self.net = ConvNet(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN, filter_K=K, pool_K=pool_K)  
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.logging   = log_dir is not None
        self.device    = device
        self.lr        = lr
        if self.logging:
            self.writer = SummaryWriter(log_dir=log_dir,flush_secs=1)
 
    def fit(self,data_loader,epochs,test_dataloader=None,verbose=False):
        """
            fits the classifier to the input data.
        
            Parameters
            ----------
            data_loader : torch-geometric dataloader
                the training dataset.
            epochs : int
                number of epochs.
            test_dataloader : torch-geometric dataloader, default=None
                the test dataset on which the model is evaluated in each epoch.
            verbose : boolean, default=False
                whether to print out loss during training.
        """    
        if self.logging:
            data= next(iter(data_loader))
            self.writer.add_graph(self.net,[data.x,data.edge_index])
        # self.scheduler = lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=0.01,step_size_up=5,mode="triangular2")

        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=1, eta_min=0.00005, last_epoch=-1)
        for epoch in range(epochs):
            self.net.train()
            self.net.to(self.device)
            total_loss = 0
            
            for batch in data_loader:
                x, edge_index, label = batch.x.to(self.device), batch.edge_index.to(self.device), batch.y.to(self.device) 
                self.optimizer.zero_grad()
                pred  = self.net(x, edge_index)
                loss  = self.criterion(pred,label)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item() * batch.num_graphs 
            total_loss /= len(data_loader.dataset)
            if verbose and epoch%(epochs//10)==0:
                print('[%d] loss: %.3f' % (epoch + 1,total_loss))

            if self.logging:
                #Save the training loss, the training accuracy and the test accuracy for tensorboard vizualisation
                self.writer.add_scalar("Training Loss",total_loss,epoch)
                accuracy_train = self.eval(data_loader,verbose=False)[0]
                self.writer.add_scalar("Accuracy on Training Dataset",accuracy_train,epoch)
                if test_dataloader is not None:
                    accuracy_test = self.eval(test_dataloader,verbose=False)[0]
                    self.writer.add_scalar("Accuracy on Test Dataset",accuracy_test,epoch)
            
                


        

    def eval(self,data_loader,verbose=False):
        """
            evaluates the model based on predictions
        
            Parameters
            ----------
            test_dataloader : torch-geometric dataloader, default=None
                the dataset on which the model is evaluated.
            verbose : boolean, default=False
                whether to print out loss during training.
            Returns
            ----------
            accuracy : float
                accuracy
            conf_mat : ndarray
                confusion matrix
            precision : fload
                weighted precision score
            recall : float
                weighted recall score
            f1_score : float
                weighted f1 score
        """  
        self.net.eval()
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in data_loader:
                x, edge_index, label = batch.x.to(self.device), batch.edge_index.to(self.device), batch.y.to('cpu')
                y_true.extend(list(label))
                outputs = self.net(x, edge_index)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to('cpu')
                y_pred.extend(list(predicted))
        accuracy, conf_mat, precision, recall, f1_score = compute_metrics(y_true, y_pred)
        if verbose:
            print('Accuracy: {:.3f}'.format(accuracy))
            print('Confusion Matrix:\n', conf_mat)
            print('Precision: {:.3f}'.format(precision))
            print('Recall: {:.3f}'.format(recall))
            print('f1_score: {:.3f}'.format(f1_score))
        return accuracy, conf_mat, precision, recall, f1_score

    def interpret(self, data_loader, n_features, n_classes):
        """
            interprets a trained model, by giving importance scores assigned to each feature regarding each class
            it uses the `IntegratedGradients` method from the package `captum` to computed class-wise feature importances 
            and then computes entropy values to get a global importance measure.
            
            Parameters
            ----------
            data_loder : torch-geometric dataloader, default=None
                the dataset on which the model is evaluated.
            n_features : int
                number of features.
            n_classes : int
                number of classes.
            
            Returns
            ----------
            ent : numpy ndarray, shape (n_features)
            
        """  
        batch = next(iter(data_loader))
        e = batch.edge_index.to(self.device).long()
        def model_forward(input):
            out = self.net(input, e)
            return out
        self.net.eval()
        importances = np.zeros((n_features, n_classes))
        for batch in data_loader:
            input = batch.x.to(self.device)
            target = batch.y.to(self.device)
            ig = IntegratedGradients(model_forward)
            attributions = ig.attribute(input, target=target)
            attributions = attributions.to('cpu').detach().numpy()
            attributions = attributions.reshape(n_features, len(target))
            attributions = normalize(attributions, axis=0, norm='l2')
#             attributions /= np.linalg.norm(attributions)
            importances[:, target.to('cpu').numpy()] += attributions
#         importances = np.e**importances
#         importances = importances / importances.max(axis=0)
#         imp = (importances.T / np.sum(importances, axis = 1)).T
#         ent = (-imp * np.log2(imp)).sum(axis = 1) / np.log2(n_classes)
#         idx = (-importances).argsort(axis=0) 
#         ent = np.min(idx, axis=1)
        return importances
