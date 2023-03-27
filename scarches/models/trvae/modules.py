import torch
import torch.nn as nn

from ._utils import one_hot_encoder

data_list = []
next_data_list = []
old_data_initial_list = []
old_data_final_list = []

class CondLayers(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_cond: int,
            bias: bool,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        if self.n_cond == 0:
            out = self.expr_L(x)
            
        else:
            expr, cond = torch.split(x, [x.shape[1] - self.n_cond, self.n_cond], dim=1)#tensor.shape[1] gives you no. cols
            out = self.expr_L(expr) + self.cond_L(cond)
        return out


class Encoder(nn.Module):
    """ScArches Encoder class. Constructs the encoder sub-network of TRVAE and CVAE. It will transform primary space
       input to means and log. variances of latent space with n_dimensions = z_dimension.

       Parameters
       ----------
       layer_sizes: List
            List of first and hidden layer sizes
       latent_dim: Integer
            Bottleneck layer (z)  size.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
       use_dr: Boolean
            If `True` dropout will applied to layers.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropput will be applied.
       num_classes: Integer
            Number of classes (conditions) the data contain. if `None` the model will be a normal VAE instead of
            conditional VAE.
    """
    def __init__(self,
                 layer_sizes: list,
                 latent_dim: int,
                 use_bn: bool,
                 use_ln: bool,
                 use_dr: bool,
                 dr_rate: float,
                 num_classes: int = None):
        super().__init__()
        self.n_classes = 0
        if num_classes is not None:
            self.n_classes = num_classes
        self.FC = None
        if len(layer_sizes) > 1:
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])): #v = [1000, 128, 128]
                                                                                            #print(v[:-1], v[1:])
                                                                                            #zipped = zip(v[:-1], v[1:])
                                                                                            #list(zipped) --> [(1000, 128), (128, 128)]
                if i == 0:
                    print("\tInput Layer in, out and cond:", in_size, out_size, self.n_classes)
                    self.FC.add_module(name="L{:d}".format(i), module=CondLayers(in_size,
                                                                                 out_size,
                                                                                 self.n_classes,
                                                                                 bias=True))
                else:
                    print("\tHidden Layer", i, "in/out:", in_size, out_size)
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=True))
                if use_bn:
                    self.FC.add_module("N{:d}".format(i), module=nn.BatchNorm1d(out_size, affine=True))
                elif use_ln:
                    self.FC.add_module("N{:d}".format(i), module=nn.LayerNorm(out_size, elementwise_affine=False))
                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if use_dr:
                    latent_layer=False
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dr_rate))
        print("\tMean/Var Layer in/out:", layer_sizes[-1], latent_dim)
        self.mean_encoder = nn.Linear(layer_sizes[-1], latent_dim)
        self.log_var_encoder = nn.Linear(layer_sizes[-1], latent_dim)
    def forward(self, x, batch=None, external_memory=0, dataset_counter=0, first_epoch=0, replay_layer=0):
        torch.autograd.set_detect_anomaly(True)
        import math
        import random
        import numpy as np
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_classes)
            x = torch.cat((x, batch), dim=-1) 

        if self.FC is not None:           
            x = self.FC(x)  
            per=1 #percentage of task i you take to concatenate later with the task i+1
            if external_memory ==1:# ONLY save the training data
                if first_epoch == 1 and replay_layer ==1: #save the training data ONLY on the first epoch
                    x=x.detach()
                    data_list.append(x) #save task i
                    if dataset_counter!=0:
                        next_data_list.append(x) # save task i+1

                else:                    
                    if dataset_counter!=0: #if task i+1, start splitting the whole data into new and old, new from task i+1, old from task i
                        new_data = x
                        new_batches = len(data_list)-len(next_data_list) 

                        if new_batches==len(data_list):
                            old_data = data_list
                        else:
                            old_data = data_list[:new_batches]

                        mb_size = x.size(0)
                        cur_sz = int(mb_size*0.2) # how much of the mini batch size will be taken by the new data 'task i+1'
                        n2inject = mb_size - cur_sz # how much of the mini batch size will be taken by the old data 'task i'

                        old_data_percent=random.sample(old_data,int(len(old_data)*per)) #Randomly get a percentage from old data 'task i'
                        cuda0 = torch.device('cuda:0')

                        #Loop over old data to make the study labels of the same length
                        for idxe, item in enumerate(old_data_percent):
                            for idxe2, item2 in enumerate(item):
                                if x.size(-1)==item2.size(-1):
                                    element = item2
                                else:
                                    item2 = torch.unsqueeze(item2,0)
                                    element = torch.cat((item2,
                                                         torch.zeros(item2.size(0),x.size(-1)-item2.size(-1),device=cuda0)),dim=-1)
                                    element = torch.squeeze(element,0)
                                old_data_initial_list.append(element)
                            catted = torch.stack(old_data_initial_list,0)

                            old_data_final_list.append(catted)

                        old_tensor=torch.cat(old_data_final_list,0)
                        del old_data_initial_list[:] #delet the list because when it is filled again, the new study labels will be of                                                            #different length
                        del old_data_final_list[:] #delet the list because when it is filled again, the new study labels will be of                                                            #different length

                        # checking if padding data is needed to fix the batch dimensions in the old_tensor
                        n_missing = old_tensor.shape[0] % mb_size
                        if n_missing > 0:
                            surplus = 1
                        else:
                            surplus = 0
                        # computing iters over old_tensor
                        old_loop = old_tensor.shape[0] // mb_size + surplus #it's like 5//2+1=2.5 and cuz //, then floor, 2+1 =3 end result
                        # padding data to fix batch dimensions
                        if n_missing > 0:
                            n_to_add = mb_size - n_missing
                            old_tensor = torch.cat((old_tensor[:n_to_add], old_tensor))


                        new_tensor=x
                        # checking if padding data is needed to fix the batch dimensions in the new_tensor
                        n_missing = new_tensor.shape[0] % mb_size
                        if n_missing > 0:
                            surplus = 1
                        else:
                            surplus = 0
                        # computing iters over new_tensor
                        new_loop = new_tensor.shape[0] // mb_size + surplus #it's like 5//2+1=2.5 and cuz //, then floor, 2+1 =3 end result
                        # checking if padding data is needed to fix the batch dimensions in the new_tensor
                        if n_missing > 0:
                            n_to_add = mb_size - n_missing
                            new_tensor = torch.cat((new_tensor[:n_to_add], new_tensor))

                        # loop over the old_tensor and new_tensor then concatenate
                        if new_loop>old_loop:                            
                            for it in range(new_loop): 
                                start_new = it * (cur_sz)
                                end_new = (it + 1) * (cur_sz)

                                start_previous = it * (n2inject)
                                end_previous = (it + 1) * (n2inject)
                                x= torch.cat((new_tensor[start_new:end_new],old_tensor[start_previous:end_previous]),0)       

                                it+=1
                                if it>old_loop: 
                                    it = random.randrange(old_loop)
                                    start_previous = it * (n2inject)
                                    end_previous = (it + 1) * (n2inject)
                                    x= torch.cat((new_tensor[start_new:end_new],old_tensor[start_previous:end_previous]),0)

                        elif new_loop<old_loop:
                            for it in range(old_loop): 
                                start_new = it * (cur_sz)
                                end_new = (it + 1) * (cur_sz)

                                start_previous = it * (n2inject)
                                end_previous = (it + 1) * (n2inject)
                                x= torch.cat((new_tensor[start_new:end_new],old_tensor[start_previous:end_previous]),0)
                                
                                it+=1
                                if it>new_loop:
                                    it = random.randrange(new_loop)
                                    start_new = it * (cur_sz)
                                    end_new = (it + 1) * (cur_sz)
                                    x= torch.cat((new_tensor[start_new:end_new],old_tensor[start_previous:end_previous]),0)

        means = self.mean_encoder(x)
        log_vars = self.log_var_encoder(x)

        return means, log_vars

    
class Decoder(nn.Module):
    """ScArches Decoder class. Constructs the decoder sub-network of TRVAE or CVAE networks. It will transform the
       constructed latent space to the previous space of data with n_dimensions = x_dimension.

       Parameters
       ----------
       layer_sizes: List
            List of hidden and last layer sizes
       latent_dim: Integer
            Bottleneck layer (z)  size.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
       use_dr: Boolean
            If `True` dropout will applied to layers.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropput will be applied.
       num_classes: Integer
            Number of classes (conditions) the data contain. if `None` the model will be a normal VAE instead of
            conditional VAE.
    """
    def __init__(self,
                 layer_sizes: list,
                 latent_dim: int,
                 recon_loss: str,
                 use_bn: bool,
                 use_ln: bool,
                 use_dr: bool,
                 dr_rate: float,
                 num_classes: int = None):
        super().__init__()
        self.use_dr = use_dr
        self.recon_loss = recon_loss
        self.n_classes = 0
        if num_classes is not None:
            self.n_classes = num_classes
        layer_sizes = [latent_dim] + layer_sizes
        print("Decoder Architecture:")
        # Create first Decoder layer
        self.FirstL = nn.Sequential()
        print("\tFirst Layer in, out and cond: ", layer_sizes[0], layer_sizes[1], self.n_classes)
        self.FirstL.add_module(name="L0", module=CondLayers(layer_sizes[0], layer_sizes[1], self.n_classes, bias=False))
        if use_bn:
            self.FirstL.add_module("N0", module=nn.BatchNorm1d(layer_sizes[1], affine=True))
        elif use_ln:
            self.FirstL.add_module("N0", module=nn.LayerNorm(layer_sizes[1], elementwise_affine=False))
        self.FirstL.add_module(name="A0", module=nn.ReLU())
        if self.use_dr:
            self.FirstL.add_module(name="D0", module=nn.Dropout(p=dr_rate))

        # Create all Decoder hidden layers
        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                if i+3 < len(layer_sizes):
                    print("\tHidden Layer", i+1, "in/out:", in_size, out_size)
                    self.HiddenL.add_module(name="L{:d}".format(i+1), module=nn.Linear(in_size, out_size, bias=False))
                    if use_bn:
                        self.HiddenL.add_module("N{:d}".format(i+1), module=nn.BatchNorm1d(out_size, affine=True))
                    elif use_ln:
                        self.HiddenL.add_module("N{:d}".format(i + 1), module=nn.LayerNorm(out_size, elementwise_affine=False))
                    self.HiddenL.add_module(name="A{:d}".format(i+1), module=nn.ReLU())
                    if self.use_dr:
                        self.HiddenL.add_module(name="D{:d}".format(i+1), module=nn.Dropout(p=dr_rate))
        else:
            self.HiddenL = None

        # Create Output Layers
        print("\tOutput Layer in/out: ", layer_sizes[-2], layer_sizes[-1], "\n")
        if self.recon_loss == "mse":
            self.recon_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.ReLU())
        if self.recon_loss == "zinb":
            # mean gamma
            self.mean_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1))
            # dropout
            self.dropout_decoder = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        if self.recon_loss == "nb":
            # mean gamma
            self.mean_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1))

    def forward(self, z, batch=None):
        # Add Condition Labels to Decoder Input
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_classes)
            z_cat = torch.cat((z, batch), dim=-1)
            dec_latent = self.FirstL(z_cat)
        else:
            dec_latent = self.FirstL(z)

        # Compute Hidden Output
        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent
        # Compute Decoder Output
        if self.recon_loss == "mse":
            recon_x = self.recon_decoder(x)
            return recon_x, dec_latent
        elif self.recon_loss == "zinb":
            dec_mean_gamma = self.mean_decoder(x)
            dec_dropout = self.dropout_decoder(x)
            return dec_mean_gamma, dec_dropout, dec_latent
        elif self.recon_loss == "nb":
            dec_mean_gamma = self.mean_decoder(x)
            return dec_mean_gamma, dec_latent
