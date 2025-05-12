import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io
import os

def predict_bird(image_input):
    # === Configuration ===
    tflite_model_path = "assets/bird_model_float32.tflite"
    
    labels_file_path = "assets/labels.txt"  # the file containing class names
    target_size = (224, 224)  # Reduced target size for smaller memory footprint

    try:
        # === Load class names from labels.txt ===
        with open(labels_file_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        # === Load TFLite model ===
        print("Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # === Load and preprocess image ===
        # print(f"Loading image from: {img_path}")
        # img = image.load_img(img_path, target_size=target_size)
        # img_array = image.img_to_array(img)
        # img_array = preprocess_input(img_array)
        # input_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)

         # === Load and preprocess image ===
        if isinstance(image_input, str) and os.path.exists(image_input):
            print(f"Loading image from file path: {image_input}")
            img = image.load_img(image_input, target_size=target_size)
        else:
            print("Loading image from in-memory stream")
            img = Image.open(io.BytesIO(image_input.read())).convert('RGB')
            img = img.resize(target_size)

        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        input_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)

        # === Run inference ===
        print("Running inference...")
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class = np.argmax(output[0])
        confidence = np.max(output[0])

        # === Output Prediction ===
        print(f"✅ Prediction: {class_names[predicted_class]} ({confidence * 100:.2f}%)")
        pre = class_names[predicted_class]
        conf = round(confidence * 100, 2)
        
        confi = float(conf)
        return pre, confi

    except Exception as e:
        print(f"Error occurred: {e}")