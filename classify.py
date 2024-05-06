import argparse
import time
import numpy as np
import cv2 
#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite

model_filepath = 'mobilenet_model_v1.tflite'
label_filepath = 'labels_mobilenet_model_v1.txt'
interpreter = tflite.Interpreter(model_path=model_filepath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
with open(label_filepath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def classify(frame):
    frame = cv2.resize(frame, (224,224))
    frame = np.expand_dims(frame, axis=0)
    interpreter.set_tensor(input_details[0]['index'], frame)
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    top_k = results.argsort()[-5:][::-1]
    for i in top_k:
        print('{:08.6f}: {} , {}'.format(float(results[i] / 255.0), labels[i]))
    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

if __name__ == "__main__" :
    imgpath = 'cat.jpg'
    frame = cv2.imread(imgpath) 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    classify(frame_rgb) 
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

'''



'''

