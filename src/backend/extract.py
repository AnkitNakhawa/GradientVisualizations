from flask import Flask, render_template
from flask_socketio import SocketIO
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os

# Get absolute path to templates directory
template_dir = os.path.abspath(os.path.join(os.getcwd(), 'templates'))
app = Flask(__name__, template_folder=template_dir)
socketio = SocketIO(app)

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

# Gradient data to send to frontend
training_data = {"nodes": [], "edges": []}

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

# Training function
def train_model():
    for epoch in range(30):  # Simulate 5 epochs
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

        # Emit the training data to the frontend
        socketio.emit('update', training_data)

        time.sleep(1)

# Route to serve the frontend
@app.route('/')
def index():
    print("Template directory:", template_dir)
    print("Current working directory:", os.getcwd())
    return render_template('index.html')

# Add training route handler
@socketio.on('start_training')
def handle_training_start():
    print("Training started")
    # Simulate training loop
    for epoch in range(5):  # Simulate 5 epochs
        # Reset training data for each iteration
        training_data["nodes"] = []
        
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
        
        # Save training data to JSON file
        with open('src/backend/training_output.json', 'w') as f:
            json.dump(training_data, f, indent=4)
        
        # Emit the updated data
        socketio.emit('training_update', training_data)
        time.sleep(1)

if __name__ == "__main__":
    socketio.run(app, debug=True)
