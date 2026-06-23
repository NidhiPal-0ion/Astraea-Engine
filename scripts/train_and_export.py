import torch
import torch.nn as nn
import torch.optim as optim
import os

class LobTCN(nn.Module):
    def __init__(self, input_size=40, hidden_size=64, num_classes=3):
        super(LobTCN, self).__init__()
        # Simulating a basic Temporal Convolutional Network for LOB prediction
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        # Sequence length is assumed to be 10 ticks: 10 * hidden_size
        self.fc = nn.Linear(hidden_size * 10, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        # Conv1d expects [batch_size, channels, seq_len]
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train_and_export():
    print("Initializing Astraea-Engine Mock TCN Model...")
    
    # 40 features (10 levels of bid/ask price and volume)
    input_size = 40
    seq_len = 10
    batch_size = 32
    num_classes = 3 # e.g., Up, Down, Stationary
    
    model = LobTCN(input_size=input_size, hidden_size=64, num_classes=num_classes)
    model.train()
    
    # Mock FI-2010 Data
    X_dummy = torch.randn(batch_size, seq_len, input_size)
    y_dummy = torch.randint(0, num_classes, (batch_size,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Running dummy training epoch...")
    for _ in range(5):
        optimizer.zero_grad()
        output = model(X_dummy)
        loss = criterion(output, y_dummy)
        loss.backward()
        optimizer.step()
        
    print(f"Dummy training complete. Final loss: {loss.item():.4f}")
    
    # Export to ONNX
    model.eval()
    
    # Create a dynamic batch size axis so TensorRT can optimize execution profiles
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    onnx_path = "model.onnx"
    print(f"Exporting ONNX graph to {onnx_path}...")
    
    # Single sample trace
    x_trace = torch.randn(1, seq_len, input_size)
    
    torch.onnx.export(
        model,
        x_trace,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print("ONNX export successful.")

if __name__ == "__main__":
    train_and_export()
