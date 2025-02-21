import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

print("Script started...")


class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(2, 512)  # Increased neurons
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)  # Additional layer
        self.fc7 = nn.Linear(16, 1)   # Output layer
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        return self.fc7(x)


def train_model(device, batch_size=1024, num_samples=50000):  
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    
    model = DeepMLP().to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002)  
    
    # Generate dataset
    x_train = torch.rand(num_samples, 2, dtype=torch.float, device=device)
    y_train = (x_train[:, 0] > 0.5).float().unsqueeze(1)  
    
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    torch.cuda.synchronize()
    
    start_time = time.time()
    
    scaler = torch.amp.GradScaler()
    
    for epoch in range(1500):  
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda"):  
                output = model(batch_x)
                loss = criterion(output, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(dataloader))
        
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/1500], Loss: {losses[-1]:.4f}')
    
    torch.cuda.synchronize()
    
    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")
    
    # Plot loss curve
    plt.plot(range(1500), losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

# Run benchmark only on GPU
if torch.cuda.is_available():
    device_gpu = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # Optimize GPU performance
    train_model(device_gpu)
else:
    print("No GPU available.")
