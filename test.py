import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="yolov5s-fp16.tflite")
interpreter.allocate_tensors()

# Load input image
input_image = cv2.imread(r"C:\Users\samee\Documents\GitHub\yolov5\data\images\crow.jpg")

original_height, original_width, _ = input_image.shape

# Resize and normalize the image
input_resized = cv2.resize(input_image, (640, 640))
input_normalized = input_resized / 255.0
input_data = np.expand_dims(input_normalized.astype(np.float32), axis=0)

# Set input tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get model output
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# Post-process output
boxes, confidences, class_ids = [], [], []
for detection in output_data:
    confidence = detection[4]  # Object confidence
    class_id = np.argmax(detection[5:])  # Class ID
    if confidence > 0.5 and class_id == 14:  # Class ID 14 is 'bird'
        x_center, y_center, width, height = detection[:4]
        x_min = int((x_center - width / 2) * original_width)
        y_min = int((y_center - height / 2) * original_height)
        x_max = int((x_center + width / 2) * original_width)
        y_max = int((y_center + height / 2) * original_height)
        boxes.append((x_min, y_min, x_max, y_max))
        confidences.append(confidence)
        class_ids.append(class_id)

# Crop and save the bird image
for i, box in enumerate(boxes):
    x_min, y_min, x_max, y_max = box
    cropped_bird = input_image[y_min:y_max, x_min:x_max]
    cv2.imwrite(f"cropped_bird_{i}.jpg", cropped_bird)
print("Detected boxes:", boxes)
print("Confidences:", confidences)
