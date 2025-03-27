import os
import time
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from denoise_model import UNet
from dataset_hd5 import IpaintDataset
from cidm import GaussianDiffusion
from draw import draw
from make_mask import add_mask
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config('config.yaml')
device = config['device']


def load_dataset(folder_path, batch_size, shuffle=True):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')]
    dataset = IpaintDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader

train_dataloader = load_dataset(config['train_data_path'], config['train_batch_size'])
test_dataloader = load_dataset(config['test_data_path'], config['test_batch_size'], shuffle=False)


def initialize_model(config, device):
    model = UNet(
        T=config['time_steps'], 
        ch=config['model']['channels'], 
        ch_mult=config['model']['channel_multiplier'], 
        attn=config['model']['attention_layers'], 
        num_res_blocks=config['model']['num_res_blocks'], 
        dropout=config['model']['dropout']
    )
    model.load_state_dict(torch.load(config['pretrained_model_path'], map_location='cpu'))
    model.to(device)
    return model

model = initialize_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
gaussian_diffusion = GaussianDiffusion(timesteps=config['time_steps'])

# 训练函数
def train_one_epoch(model, dataloader, optimizer, diffusion, device, epoch, global_step):
    model.train()
    total_loss = 0.0
    for step, images in enumerate(dataloader):
        optimizer.zero_grad()
        images = images.float().to(device)
        batch_size = images.shape[0]
        t = torch.randint(0, config['time_steps'], (batch_size,), device=device).long()
        loss = diffusion.train_losses(model, images, t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        global_step += 1

        if step % config['log_interval'] == 0:
            avg_loss = total_loss / (step + 1)
            logging.info(f"Epoch {epoch}, Step {step}/{len(dataloader)}, Loss: {avg_loss:.4f}")
    return total_loss / len(dataloader), global_step


def visualize_diffusion(model, diffusion, images, epoch, output_dir):
    t = torch.randint(0, config['time_steps'], (1,), device=device).long()
    x_t = diffusion.q_sample(images, t)
    predicted_noise = model(x_t, t)
    x0 = diffusion.predict_start_from_noise(x_t, t, predicted_noise)[0]
    x0 = x0.reshape(256, 256).detach().cpu().numpy() * 80
    images = images[0].reshape(256, 256).detach().cpu().numpy() * 80
    draw(images, os.path.join(output_dir, f'{epoch}_truth.png'))
    draw(x0, os.path.join(output_dir, f'{epoch}_pred_x0.png'))


def save_model(model, epoch, output_dir):
    model_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved at {model_path}")

def train_model():
    global_step = 0
    for epoch in range(config['epochs']):
        avg_loss, global_step = train_one_epoch(model, train_dataloader, optimizer, gaussian_diffusion, device, epoch, global_step)
        logging.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        if epoch % config['save_interval'] == 0:
            visualize_diffusion(model, gaussian_diffusion, next(iter(train_dataloader)).float().to(device), epoch, config['output_dir'])
            save_model(model, epoch, config['output_dir'])

if __name__ == "__main__":
    train_model()
    
    example_config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'train_data_path': '/path/to/train/data',
        'test_data_path': '/path/to/test/data',
        'train_batch_size': 16,
        'test_batch_size': 16,
        'time_steps': 1000,
        'learning_rate': 1e-4,
        'epochs': 50,
        'log_interval': 10,
        'save_interval': 5,
        'output_dir': './output',
        'pretrained_model_path': './pretrained_model.pth',
        'model': {
            'channels': 64,
            'channel_multiplier': [1, 2, 4, 8],
            'attention_layers': [1],
            'num_res_blocks': 2,
            'dropout': 0.1
        }
    }

    
    with open('diffusion_config.yaml', 'w') as f:
        yaml.dump(example_config, f)
    logging.info("Diffusion configuration saved to 'diffusion_config.yaml'")
