from flask import Flask, render_template
from flask_socketio import SocketIO
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

template_dir = os.path.abspath(os.path.join(os.getcwd(), 'templates'))
app = Flask(__name__, template_folder=template_dir)
# The async_mode can be 'eventlet', 'gevent', or None (auto-detect).
# If you have eventlet installed, you can set socketio = SocketIO(app, async_mode='eventlet')
socketio = SocketIO(app, cors_allowed_origins="*")

# ----------------------------------------------------------------
# Example multi-layer model. Change or expand as needed.
# ----------------------------------------------------------------
class MultiLayerModel(nn.Module):
    def __init__(self):
        super(MultiLayerModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

model = MultiLayerModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# We'll send this to the front-end
training_data = {"nodes": [], "edges": []}

# 1) Gather all Linear layers
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        linear_layers.append(name)

# 2) We'll keep a dict to store gradient norms
grad_dict = {}

# Hook to capture each Linear layer's gradient
def backward_hook(module, grad_input, grad_output):
    if getattr(module, "weight", None) is not None and module.weight.grad is not None:
        # Find which layer name this module corresponds to
        for name, m in model.named_modules():
            if m is module:
                grad_dict[name] = module.weight.grad.norm().item()
                break

# 3) Register the hook on all linear layers
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.register_full_backward_hook(backward_hook)

def train_model(num_epochs):
    """
    Run training for num_epochs. After each epoch:
      - The backward hook has stored gradient norms in grad_dict
      - We build the 'nodes' and 'edges'
      - We emit 'training_update' so the front-end can re-render
      - We use socketio.sleep(...) to avoid blocking
    """
    for epoch in range(num_epochs):
        # Clear gradient data
        grad_dict.clear()

        # Standard training step
        optimizer.zero_grad()
        x = torch.randn(1, 4)
        target = torch.randn(1, 1)
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Build training_data from grad_dict
        # (Nodes: each layer with gradient)
        training_data["nodes"] = [
            {
                "id": layer_name,
                "gradient_norm": grad_dict.get(layer_name, 0.0)
            }
            for layer_name in linear_layers
        ]

        # Build edges between consecutive linear layers
        training_data["edges"] = []
        for i in range(len(linear_layers) - 1):
            source = linear_layers[i]
            target = linear_layers[i + 1]
            target_grad = grad_dict.get(target, 0.0)
            training_data["edges"].append({
                "source": source,
                "target": target,
                "gradient_norm": target_grad
            })

        # Optionally save to JSON (this will overwrite each epoch)
        with open('src/backend/training_output.json', 'w') as f:
            json.dump(training_data, f, indent=4)

        # Emit the data for this epoch
        socketio.emit('training_update', training_data)
        print(f"Epoch {epoch+1}/{num_epochs} complete. Emitted training_update.")

        # Yield to let Socket.IO push the update
        socketio.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_training')
def handle_training_start():
    print("Received 'start_training'.")
    train_model(num_epochs=30)
    print("Training completed.")

if __name__ == "__main__":
    print("Running Flask-SocketIO on port 5000...")
    socketio.run(app, debug=True, port=5000)
