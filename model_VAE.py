import torch
import torch.nn as nn
import torch.nn.functional as F

#Variational Autoencoder

#input data -> encode to hidden representation -> sample from latent space with a speciifc mean and variance
#  -> parametrization trick -> decode to output data
class VariationalAutoEncoder(nn.Module):
    def __init__(self,input_dim=30,hidden_dim=200,z_dim=10):
        super().__init__()
        
        #Encoder
        self.input_2hid=nn.Linear(input_dim,hidden_dim)
        self.hid_2_mean=nn.Linear(hidden_dim,z_dim)
        self.hid_2_sigma=nn.Linear(hidden_dim,z_dim)
        

        #in loss function we want the two linear layers to become standard gaussian so that the latent space is gaussian
        #this is done by minimizing the KL divergence between the two distributions

        #Decoder
        self.z_2_hid_2_hid=nn.Linear(z_dim,hidden_dim)
        self.hid_2_output=nn.Linear(hidden_dim,input_dim)


    #Encoder
    def encode(self, x):
        #q_phi(z given x)
        #firt layer
        hid=self.input_2hid(x)
        #second layer (relu)
        hid=F.relu(hid)
        #mean and variance
        mu=self.hid_2_mean(hid)

        sigma=self.hid_2_sigma(hid)
        return mu,sigma
    def reparametrize(self,mu,sigma):
        std=torch.exp(0.5*sigma) #this ensures that the variance is positive
        eps=torch.randn_like(std)
        return mu+eps*std

    def decode(self,z):
        #p_theta(x given z)
        #recunstruct the data
        hid=self.z_2_hid_2_hid(z)
        #apply relu
        hid=F.relu(hid)
        out=self.hid_2_output(hid)
        return out

    def forward(self,x):
        mu,sigma=self.encode(x)
        x_reparametrized=self.reparametrize(mu,sigma)
        x_reconstructed=self.decode(x_reparametrized)
        return x_reconstructed,mu,sigma
    

def vae_loss_function(reconstructed_x, x, mu, sigma):
    # Reconstruction loss (using MSE)
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')  # or reduction='mean'

    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    return recon_loss + kl_divergence

# function to calculate the reconstruction error 
def reconstruction_error(model, data_loader, device):
    model.eval()
    total_error = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            recon_batch, _, _ = model(batch)
            total_error += F.mse_loss(recon_batch, batch, reduction='sum').item()
    return total_error / len(data_loader.dataset)