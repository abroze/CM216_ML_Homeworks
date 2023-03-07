
# Importing the libraries
import torch
import torch.nn as nn
from torchsummary import summary

# ==============================================================================
# Deep Neural Network (Autoencoder)
# ==============================================================================
class FCN(nn.Module):
  def __init__(self, dim):
    super(FCN, self).__init__()
    self.dim = dim
    self.down_scale = nn.Sequential(
              nn.Conv2d(1, self.dim, 4, 2, 1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm2d(self.dim),
              nn.Conv2d(self.dim, 2 * self.dim, 4, 2, 1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm2d(2 * self.dim),
              nn.Conv2d(2 * self.dim, 4 * self.dim, 4, 2, 1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm2d(4 * self.dim),
              nn.Conv2d(4 * self.dim, 8 * self.dim, 4, 2, 1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm2d(8 * self.dim)
          )    
    self.up_scale = nn.Sequential(
              nn.ConvTranspose2d(8 * self.dim, 4 * self.dim, 4, 2, 1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm2d(4 * self.dim),
              nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 4, 2, 1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm2d(2 * self.dim),
              nn.ConvTranspose2d(2 * self.dim, self.dim, 4, 2, 1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm2d(self.dim),
              nn.ConvTranspose2d(self.dim, 1, 4, 2, 1),
              nn.LeakyReLU(0.2)
          )
  

  def forward(self, x):
    x = self.down_scale(x)
    output = self.up_scale(x)
    return output

#summary(model, (1, 32, 32))