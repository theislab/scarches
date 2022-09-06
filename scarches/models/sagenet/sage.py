from .utils import *
from .classifier import *
from .model import *
from os import listdir
import numpy as np
import anndata
import re

class sage():
    """
        A `sagenet` object.
        
        Parameters
        ----------
        device : str, default = 'cpu'
            the processing unit to be used in the classifiers (gpu or cpu).
    """

    def __init__(self, device='cpu'):
        self.models    = {}
        self.adjs      = {}
        inf_genes = None
        self.num_refs = 0
        self.device = device 

    def train(self, 
        adata, 
        tag = None,
        comm_columns = 'class_',
        classifier   = 'TransformerConv',
        num_workers  = 0,
        batch_size   = 32,
        epochs       = 10,
        n_genes      = 10,
        verbose      = False,
        importance   = False,
        to_return    = False):
        """Trains new classifiers on a reference dataset.

            Parameters
            ----------
            adata : `AnnData`
                The annotated data matrix of shape `n_obs × n_vars` to be used as the spatial reference. Rows correspond to cells (or spots) and columns to genes.  
            tag : str, default = `None`
                The tag to be used for storing the trained models and the outputs in the `sagenet` object.
            classifier : str, default = `'TransformerConv'`
                The type of classifier to be passed to `sagenet.Classifier()`
            comm_columns : list of str, `'class_'`
                The columns in `adata.obs` to be used as spatial partitions.
            num_workers : int
                Non-negative. Number of workers to be passed to `torch_geometric.data.DataLoader`.
            epochs : int
                number of epochs.
            verbose : boolean, default=False
                whether to print out loss during training.

             Return
            ------
            Returns nothing.

            Notes
            -----
            Trains the models and adds them to `.models` dictionery of the `sagenet` object.
            Also adds a new key `{tag}_entropy` to `.var` from `adata` which contains  the entropy values as the importance score corresponding to each gene.
        """    
        ind = np.where(np.sum(adata.varm['adj'], axis=1) == 0)[0]
        ents = np.ones(adata.var.shape[0]) * 1000000
#         ents = np.zeros(adata.var.shape[0])
        self.num_refs += 1

        if tag is None:
            tag = 'ref' + str(self.num_refs)

        for comm in comm_columns:
            data_loader = get_dataloader(
                graph       = adata.varm['adj'].toarray(), 
                X           = adata.X, y = adata.obs[comm].values.astype('long'), 
                batch_size  = batch_size,
                shuffle     = True, 
                num_workers = num_workers
            )

            clf = Classifier(
                n_features   = adata.shape[1],
                n_classes    = (np.max(adata.obs[comm].values.astype('long'))+1),
                n_hidden_GNN = [8],
                dropout_FC   = 0.2,
                dropout_GNN  = 0.3,
                classifier   = classifier, 
                lr           = 0.001,
                momentum     = 0.9,
                device       = self.device
            ) 

            clf.fit(data_loader, epochs = epochs, test_dataloader=None,verbose=verbose)
            if importance:
                imp = clf.interpret(data_loader, n_features=adata.shape[1], n_classes=(np.max(adata.obs[comm].values.astype('long'))+1))
                idx = (-abs(imp)).argsort(axis=0) 
                imp = np.min(idx, axis=1) 
    #             imp += imp
                np.put(imp, ind, 1000000)
                ents = np.minimum(ents, imp)
            
#             imp = np.min(idx, axis=1)
#             ents = np.minimum(ents, imp)
            self.models['_'.join([tag, comm])] = clf.net
            self.adjs['_'.join([tag, comm])] = adata.varm['adj'].toarray()
        if importance:
            if not to_return:
                save_adata(adata, attr='var', key='_'.join([tag, 'importance']), data=ents)
            else:
                return(ents)
#         return ents


    def load_query_data(self, adata_q, to_return = False):
        """Maps a query dataset to space using the trained models on the spatial reference(s).

            Parameters
            ----------
            adata : `AnnData`
                The annotated data matrix of shape `n_obs × n_vars` to be used as the query. Rows correspond to cells (or spots) and columns to genes.  

            Return
            ------
            Returns nothing.

            Notes
            -----
            * Adds new key(s) `pred_{tag}_{partitioning_name}` to `.obs` from `adata` which contains the predicted partition for partitioning `{partitioning_name}`, trained by model `{tag}`.
            * Adds new key(s) `ent_{tag}_{partitioning_name}` to `.obs` from `adata` which contains the uncertainity in prediction for partitioning `{partitioning_name}`, trained by model `{tag}`.
            * Adds a new key `distmap` to `.obsm` from `adata` which is a sparse matrix of size `n_obs × n_obs` containing the reconstructed cell-to-cell spatial distance.
        """    
        dist_mat = np.zeros((adata_q.shape[0], adata_q.shape[0]))
        for tag in self.models.keys():
            self.models[tag].eval()
            i = 0
            adata_q.obs['class_'] = 0
            data_loader = get_dataloader(
                graph       = self.adjs[tag], 
                X           = adata_q.X, y = adata_q.obs['class_'].values.astype('long'), #TODO: fix this
                batch_size  = 1,
                shuffle     = False, 
                num_workers = 0
            )
            with torch.no_grad():
                for batch in data_loader:
                    x, edge_index = batch.x.to(self.device), batch.edge_index.to(self.device)
                    outputs = self.models[tag](x, edge_index)
                    predicted = outputs.data.to('cpu').detach().numpy()
                    i += 1
                    if i == 1:
                        n_classes = predicted.shape[1]
                        y_pred = np.empty((0, n_classes))
                    y_pred = np.concatenate((y_pred, predicted), axis=0)
            
            y_pred = np.exp(y_pred)
            y_pred = (y_pred.T / y_pred.T.sum(0)).T
            save_adata(adata_q, attr='obs', key='_'.join(['pred', tag]), data = np.argmax(y_pred, axis=1))
            temp = (-y_pred * np.log2(y_pred)).sum(axis = 1)
            # adata_q.obs['_'.join(['ent', tag])] = np.array(temp) / np.log2(n_classes)
            save_adata(adata_q, attr='obs', key='_'.join(['ent', tag]), data = (np.array(temp) / np.log2(n_classes)))
            y_pred_1 = (multinomial_rvs(1, y_pred).T * np.array(adata_q.obs['_'.join(['ent', tag])])).T
            y_pred_2 = (y_pred.T * (1-np.array(adata_q.obs['_'.join(['ent', tag])]))).T
            y_pred_final = y_pred_1 + y_pred_2
            kl_d = kullback_leibler_divergence(y_pred_final)
            kl_d = kl_d + kl_d.T
            kl_d /= np.linalg.norm(kl_d, 'fro')
            dist_mat += kl_d
        if not to_return:
            save_adata(adata_q, attr='obsm', key='dist_map', data=dist_mat)
        else:
            return(dist_mat)

    def save(self, tag, dir='.'):
        """Saves a single trained model.

            Parameters
            ----------
            tag : str
                Name of the trained model to be saved.
            dir : dir, defult=`'.'`
                The saving directory.
        """    
        path = os.path.join(dir, tag) + '.pickle'
        torch.save(self.models[tag], path)

    def load(self, tag, dir='.'):
        """Loads a single pre-trained model.

            Parameters
            ----------
            tag : str
                Name of the trained model to be stored in the `sagenet` object.
            dir : dir, defult=`'.'`
                The input directory.
        """    
        path = os.path.join(dir, tag) + '.pickle'
        self.models[tag] = torch.load(path)

    def save_as_folder(self, dir='.'):
        """Saves all trained models stored in the `sagenet` object as a folder.

            Parameters
            ----------
            dir : dir, defult=`'.'`
                The saving directory.
        """   
        for tag in self.models.keys():
            self.save(tag, dir)
            adj_path = os.path.join(dir, tag) + '.h5ad'
            adj_adata = anndata.AnnData(X = self.adjs[tag])
            adj_adata.write(filename=adj_path)

    def load_from_folder(self, dir='.'):
        """Loads pre-trained models from a directory.

            Parameters
            ----------
            dir : dir, defult=`'.'`
                The input directory.
        """
        model_files = [f for f in listdir(dir) if re.search(r".pickle$", f)]
        for m in model_files:
            tag = re.sub(r'.pickle', '', m)
            model_path = os.path.join(dir, tag) + '.pickle'
            adj_path = os.path.join(dir, tag) + '.h5ad'
            self.models[tag] = torch.load(model_path)
            self.adjs[tag] = anndata.read_h5ad(adj_path).X


