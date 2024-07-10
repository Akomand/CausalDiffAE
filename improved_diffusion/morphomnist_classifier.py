from torch.optim import Adam
#from germancredit.data.meta_data import attrs
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
import torch.nn.functional as F
import sys
sys.path.append("../../")
from image_datasets import MorphoMNISTLike, get_dataloader_pendulum
from improved_diffusion.nn import GaussianConvEncoder



# class Classifier(nn.Module):
#     def __init__(self, in_channel, in_dim, h_dim, out_dim):
#         self.in_channel = in_channel
        
#         self.in_dim = in_dim
#         self.h_dim = h_dim
#         self.out_dim = out_dim
        
#         self.net = 
        
#     def forward(self, x):
#         pass


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, gpu_id=0, dataset="morphomnist", save_every=None):
        
        self.gpu_id = gpu_id
        self.save_every = save_every
        
        # self.model = model.to(gpu_id)
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.dataset = dataset
        self.model = model.to('cuda:0')
        # self.model = DDP(self.model, device_ids=[gpu_id], find_unused_parameters=True)


    def train_one_epoch(self):
        total_loss = 0
        # loss_ema = None
        for batch_idx, data in enumerate(self.train_loader):        
            if self.dataset == "pendulum":
                X, label_dict = data
                t = label_dict["c"][:, 3].unsqueeze(1)
                # print(t.shape)
                # exit(0)
            else:
                X = data[0]
                c = data[1]

            self.optim.zero_grad()

            X = X.to('cuda:0')
            out = self.model(X)
            
            loss = nn.MSELoss()(out, t.type(torch.float32).view(-1, 1).to("cuda:0"))
            
            loss.backward()

            self.optim.step()

            total_loss += loss

        m = len(self.train_loader)
        avg_train_loss = total_loss / m

        return avg_train_loss


    def validate_one_epoch(self):
        total_loss_val = 0

        for val_batch_idx, data in enumerate(self.val_loader):
            if self.dataset == "pendulum":
                X, label_dict = data
                t = label_dict["c"][:, 3].unsqueeze(1)
            else:
                X = data[0]
                y = data[1]

            X = X.to('cuda:0')
            out = self.model(X)
            
            loss_val = nn.MSELoss()(out, t.type(torch.float32).view(-1, 1).to("cuda:0"))
            

            total_loss_val += loss_val.item()


        n = len(self.val_loader)
        avg_val_loss = total_loss_val / n

        return avg_val_loss

    def train(self):
        best_loss = 1000000
        for epoch in range(100):

            self.model.train()
            # self.train_loader.sampler.set_epoch(epoch)
            avg_train_loss = self.train_one_epoch()

            self.model.eval()
            avg_val_loss = self.validate_one_epoch()

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                model_path = 'classifier_thickness_best.pth'

                if self.gpu_id == 0: # save from rank 0 process
                    self.save_checkpoint(model_path)

            if epoch % 1 == 0:
                print("Epoch " + str(epoch)+':loss:'+str(avg_train_loss) + ', vloss:'+str(avg_val_loss))
    
    def linear_scheduler(self, step, total_steps, initial, final):
        """Linear scheduler"""

        if step >= total_steps:
            return final
        if step <= 0:
            return initial
        if total_steps <= 1:
            return final

        t = step / (total_steps - 1)
        return (1.0 - t) * initial + t * final

    def save_checkpoint(self, model_path):
        # torch.save(self.model.state_dict(), f'../results/morphomnist/classifier/{model_path}')
        torch.save(self.model.state_dict(), f'../results/morphomnist/classifier/{model_path}')

    def config_optimizer(self):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    


#just test code for the classifiers
if __name__ == "__main__":

    model = GaussianConvEncoder(in_channels=4, latent_dim=512, num_vars=4)
    model.to("cuda:0")
    
    # init dataset
    train_loader = get_dataloader_pendulum(path="../datasets/pendulum", batch_size=64, split_set="train", shard=0, num_shards=1)
    val_loader = get_dataloader_pendulum(path="../datasets/pendulum", batch_size=64, split_set="test", shard=0, num_shards=1)


    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(model=model, optimizer=optim, train_loader=train_loader, val_loader=val_loader)
    trainer.train()