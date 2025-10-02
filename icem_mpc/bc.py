import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pickle
import os
import glob
import argparse

writer = SummaryWriter('runs/fashion_mnist_experiment_1')

class BC(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes = [32, 32], dropout_rate=0.1):
        super(BC, self).__init__()
        layers = []
        prev_size = obs_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, act_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
def train_behavior_cloning(model, expert_observations, expert_actions, epochs=1000, batch_size=64, lr=1e-3, shed=[[300, 600], .5]):

    device = torch.device('cuda')
    model = model.to(device)

    # Преобразуем данные в тензоры PyTorch
    expert_obs_tensor = torch.FloatTensor(expert_observations).to(device)
    expert_acts_tensor = torch.FloatTensor(expert_actions).to(device) 
    
    # Создаем Dataset и DataLoader для батчей
    dataset = TensorDataset(expert_obs_tensor, expert_acts_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    #scheduler
    scheduler = MultiStepLR(optimizer, milestones=shed[0], gamma=shed[1])
    
    # Цикл обучения
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        
        for batch_idx, (batch_obs, batch_acts) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            predicted_actions = model.forward(batch_obs)
            
            # Вычисляем потери
            loss = criterion(predicted_actions, batch_acts.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if batch_idx == 0:
                max_loss = loss.item()
                min_loss = loss.item()
            else:
                max_loss = max(loss.item(), max_loss)
                min_loss = min(loss.item(), min_loss)
            total_loss += loss.item()
            num_batches += 1
            # writer.add_scalar('Training/Loss per batch', loss.item(), epoch * len(dataloader) + batch_idx)
        
        avg_loss = total_loss / num_batches
        writer.add_scalar('train_loss', avg_loss, epoch)
        writer.add_scalar('margin_up', max_loss, epoch)
        writer.add_scalar('margin_down', min_loss, epoch)
        layout = {'metrics': {'loss_epoch_margin': ['Margin', ['train_loss', 'margin_up', 'margin_down']]},
                    }
        writer.add_custom_scalars(layout)
        if epoch % 10 == 0:
            print(f"Эпоха {epoch}, Средние потери: {avg_loss:.4f}")
            print(lr)
    
    print("Обучение Behavior Cloning завершено!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_name', default='mobile')
    args = parser.parse_args()

    torch.seed = 42
    dataset = []
    hand_name = "shadow_dexee"
    camera_name = args.camera_name
    folder = "experts_traj_" + hand_name + "/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03/valid_traj"
    filenames = [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*_' + camera_name + '.pkl'))]

    for file_name in filenames:
        with open(file_name, 'rb') as f:
            dataset.append(pickle.load(f))
    model =  BC(4800, 18)
    # model.load_state_dict(torch.load('model_weights.pth'))
    
    observations = []
    actions = []
    for data in dataset:
        for data_i in data:
            observations.append(data['observation'])
            actions.append(data['action'])

    observations = np.concat([*observations])
    actions = np.concat([*actions])
    print(np.unique(observations))
    model = train_behavior_cloning(model, expert_observations=observations, expert_actions=actions)
    torch.save(model.state_dict(), 'model_weights_' + camera_name + '.pth')