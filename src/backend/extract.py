from flask import Flask, render_template
from flask_socketio import SocketIO
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import statistics  # For median calculation

template_dir = os.path.abspath(os.path.join(os.getcwd(), 'templates'))
app = Flask(__name__, template_folder=template_dir)
socketio = SocketIO(app, cors_allowed_origins="*")

# ----------------------------------------------------------------
# A 10-layer model to demonstrate gradient behavior
# ----------------------------------------------------------------
class MultiLayerModel(nn.Module):
    def __init__(self):
        super(MultiLayerModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 32)
        self.fc9 = nn.Linear(32, 16)
        self.fc10 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = self.fc10(x)
        return x

model = MultiLayerModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Data structure sent to the client each epoch
training_data = {"nodes": [], "edges": []}

# Gather all nn.Linear layers
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        linear_layers.append(name)

# Dictionary to store {layer_name -> gradient_norm}
grad_dict = {}

# Hook to capture each Linear layer's gradient
def backward_hook(module, grad_input, grad_output):
    if getattr(module, "weight", None) is not None and module.weight.grad is not None:
        # Identify the layer name
        for name, m in model.named_modules():
            if m is module:
                grad_dict[name] = module.weight.grad.norm().item()
                break

# Register hooks on all Linear layers
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.register_full_backward_hook(backward_hook)

def train_model(num_epochs=30):
    """
    Training loop with dynamic threshold-based coloring:
      - min_threshold = 0.1 * median_grad
      - max_threshold = 2.0 * median_grad
      => if grad < min_threshold => red
         if grad > max_threshold => green
         else => grey
    """
    for epoch in range(num_epochs):
        grad_dict.clear()

        # Use a larger batch to create more variability
        optimizer.zero_grad()
        x = torch.randn(32, 4)      # 32 samples, 4 features
        target = torch.randn(32, 1) # 32 samples, 1 target
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Build node data
        training_data["nodes"] = [
            {"id": lname, "gradient_norm": grad_dict.get(lname, 0.0)}
            for lname in linear_layers
        ]

        # Build edges list
        edges_list = []
        for i in range(len(linear_layers) - 1):
            source = linear_layers[i]
            target = linear_layers[i + 1]
            grad_value = grad_dict.get(target, 0.0)  # gradient of target layer
            edges_list.append({
                "source": source,
                "target": target,
                "gradient_norm": grad_value
            })

        # Compute median among all edges
        if edges_list:
            edge_gradients = [e["gradient_norm"] for e in edges_list]
            median_grad = statistics.median(edge_gradients)
        else:
            median_grad = 0.0

        # Thresholds
        min_threshold = 0.1 * median_grad
        max_threshold = 2.0 * median_grad

        # Assign colors based on thresholds
        for e in edges_list:
            g = e["gradient_norm"]
            if g < min_threshold:
                e["color"] = "red"
            elif g > max_threshold:
                e["color"] = "green"
            else:
                e["color"] = "grey"

        training_data["edges"] = edges_list

        # Optional: Save to JSON
        with open('src/backend/training_output.json', 'w') as f:
            json.dump(training_data, f, indent=4)

        # Emit
        socketio.emit('training_update', training_data)
        print(f"Epoch {epoch+1}/{num_epochs} | Median={median_grad:.4f},"
              f" min={min_threshold:.4f}, max={max_threshold:.4f}")
        socketio.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_training')
def handle_training_start():
    print("Received 'start_training' from client.")
    train_model(num_epochs=30)  # or however many epochs you like
    print("Training completed.")

if __name__ == "__main__":
    print("Running Flask-SocketIO on port 5000...")
    socketio.run(app, debug=True, port=5000)
