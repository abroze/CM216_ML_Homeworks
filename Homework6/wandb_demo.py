# Import libraries
import os
import pprint
import numpy as np
import wandb
wandb.login(key="")       # Use your unique API key here! You may find it in your wandb account settings.
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import *
from models import FCN
# Ensure reproducibility
torch.backends.cudnn.deterministic = True
seed_no = 108
np.random.seed(hash("improves reproducibility") % seed_no)
torch.manual_seed(hash("by removing stochasticity") % seed_no)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed_no)
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)







# ==============================================================================
# Hyperparameters and Data loader
# ==============================================================================
# Loading the data
n_data = 4608
m_field = np.load("demo_data/k_fullfield_diffusion.npy")[:n_data, :, :]
u_field = np.load("demo_data/u_fullfield_diffusion.npy")[:n_data, :, :]

# Normalize the data between -1 and 1
m_field = ((m_field - m_field.min())/(m_field.max()-m_field.min()))*2 - 1.
u_field = ((u_field - u_field.min())/(u_field.max()-u_field.min()))*2 - 1.

# Hyperparameters
batch_size = 256
#lr = 0.01
num_epochs = 40
log_freq = int(0.1 * num_epochs)
n_train = int(0.9 * m_field.shape[0])

# Data loader
train_m_field = torch.from_numpy(m_field[:n_train, :, :]).float().to(device)
train_u_field = torch.from_numpy(u_field[:n_train, :, :]).float().to(device)
val_m_field = torch.from_numpy(m_field[n_train:, :, :]).float().to(device)
val_u_field = torch.from_numpy(u_field[n_train:, :, :]).float().to(device)
train_dataset = torch.utils.data.TensorDataset(train_m_field, train_u_field)
test_dataset = torch.utils.data.TensorDataset(val_m_field, val_u_field)

#wandb hyperparameter dictionary
sweep_configuration = {
    "method": "grid",
    "name": "grid_search",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": 
    {
        "n_channels": {"values": [4, 16, 128]},
        "reg_param": {"values": [0.0]},
        "lr": {"values": [0.005]},
        "batch_size": {"values": [8, 256, 1024]}
     }
}
pprint.pprint(sweep_configuration)
project_name = "cme216_wandb_demo"
group_name = "grid_search_hw"
sweep_id = wandb.sweep(sweep_configuration, project=project_name)



import time
t1 = time.time()
# ==============================================================================
# Training
# ==============================================================================
# Train the model
def train(config=None):
    # Initialize the new wandb run
    wandb.init(config=config, project=project_name, group=group_name) 
    config = wandb.config
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=config.batch_size,
                                                shuffle=True)
    total_step = len(train_loader)
    loss_list = []


    # Model, Loss, and Optimizer
    model = FCN(config.n_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.reg_param)
    for epoch in range(num_epochs):
        for i, (train_x, train_y) in enumerate(train_loader):
            # Run the forward pass
            model.train()
            output = model(train_x.unsqueeze(1))
            loss = criterion(output, train_y.unsqueeze(1))
            loss_list.append(loss.item())
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % log_freq == 0:
            # Calculate the validation loss
            model.eval()
            with torch.no_grad():
                val_u_field_pred = model(val_m_field.unsqueeze(1))                    
                val_loss = criterion(val_u_field_pred, val_u_field.unsqueeze(1))
            
            plot_perm_and_temp(val_m_field.detach().cpu().numpy().squeeze(), 
                                val_u_field.detach().cpu().numpy().squeeze(), 
                                val_u_field_pred.detach().cpu().numpy().squeeze(), epoch)
            diff_ = (val_u_field_pred - val_u_field.unsqueeze(1)).detach().cpu().numpy().squeeze()
            diff_vec = np.reshape(diff_, (diff_.shape[0], -1))
            val_l2_pt_error = np.mean(np.linalg.norm(diff_vec, axis=1) / np.linalg.norm(np.reshape(val_u_field.detach().cpu().numpy(), (val_u_field.shape[0], -1)), axis=1), axis=0) * 100
            
            wandb.log({"val_loss": val_loss.item(), "train_loss": loss.item(), "val_rel_error_pt": val_l2_pt_error, "epoch": epoch})
            print (f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], \
                    Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, \
                    Val. error (in %) = {val_l2_pt_error:.2f}%")

    # Save the model checkpoint (optional)
    save_path = os.path.join(wandb.run.dir, "model.ckpt")
    torch.save(model.state_dict(), save_path)

wandb.agent(sweep_id, train)
t2 = time.time()
print(f"Total time taken: {t2-t1}")
wandb.finish()