import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify
from threading import Thread
import time
import json

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Training setup
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Dictionary to store gradient and weight information
training_data = {
    "nodes": [],
    "edges": []
}

# Hook to capture gradient norms
def backward_hook(module, grad_input, grad_output):
    if hasattr(module, 'weight') and module.weight.grad is not None:
        training_data["nodes"].append({
            "id": module.__class__.__name__,
            "gradient_norm": module.weight.grad.norm().item()
        })

# Register hooks
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.register_full_backward_hook(backward_hook)

# Add this function to save the data
def save_training_data():
    with open('training_output.json', 'w') as f:
        json.dump(training_data, f, indent=4)

# Modify the train_model function to save after training
def train_model():
    for epoch in range(5):  # Simulate 5 epochs
        optimizer.zero_grad()
        x = torch.randn(1, 4)
        target = torch.randn(1, 2)
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Capture weight norms as edges
        training_data["edges"] = [
            {"source": "fc1", "target": "fc2", "gradient_norm": model.fc2.weight.grad.norm().item()}
        ]
        time.sleep(1)
    
    # Save the data after training
    save_training_data()

# Start training in a separate thread
training_thread = Thread(target=train_model)
training_thread.start()

# Run the training
if __name__ == "__main__":
    train_model()


