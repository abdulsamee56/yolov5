import torch
import cv2
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes

# Load the model
model = DetectMultiBackend('yolov5s.torchscript')
model.eval()

# Load the input image
image_path = "data/images/test1.png"  # Replace with your image path
input_image = cv2.imread(image_path)
original_height, original_width, _ = input_image.shape

# Resize and normalize the image
input_resized = cv2.resize(input_image, (640, 640))
input_normalized = input_resized / 255.0  # Normalize to 0-1
input_tensor = torch.from_numpy(input_normalized).permute(2, 0, 1).unsqueeze(0).float()

# Run inference
with torch.no_grad():
    outputs = model(input_tensor)

# Apply NMS with a lower confidence threshold
predictions = non_max_suppression(outputs, conf_thres=0.25, iou_thres=0.5)

if predictions[0] is None:
    print("No detections were made.")
else:
    print(f"Number of detections: {len(predictions[0])}")

    # Process predictions
    for i, detection in enumerate(predictions[0]):  # Batch size is 1
        x_min, y_min, x_max, y_max, confidence, class_id = detection
        class_id = int(class_id.item())
        confidence = confidence.item()
        print(f"Detection {i}: Class ID {class_id}, Confidence {confidence:.2f}")

        # For now, save all detections
        # Convert to numpy array and reshape for scale_boxes
        boxes = detection[:4].view(1, 4)
        # Scale boxes from model input size to original image size
        boxes = scale_boxes(input_tensor.shape[2:], boxes, input_image.shape).round()
        x_min, y_min, x_max, y_max = boxes[0].tolist()
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        # Crop and save detected object
        cropped_image = input_image[y_min:y_max, x_min:x_max]
        output_path = f"cropped_object_{i}_class_{class_id}.jpg"
        cv2.imwrite(output_path, cropped_image)
        print(f"Saved cropped image: {output_path}")

print("Detection complete.")
