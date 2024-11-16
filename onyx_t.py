import torch
from models.common import DetectMultiBackend  # Import YOLOv5 model class

# Load the YOLOv5 model
model = DetectMultiBackend("yolov5s.pt")  # Use the YOLOv5 model loading utility
model.model.eval()  # Set the model to evaluation mode

# Convert the model to TorchScript
scripted_model = torch.jit.script(model.model)  # Convert the underlying model to TorchScript
scripted_model.save("yolov5s_scripted.pt")  # Save the TorchScript model
