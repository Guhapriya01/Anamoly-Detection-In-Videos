from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

# Load your trained LRCN model
model = load_model("Suspicious_Human_Activity_Detection_LRCN_Model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_anomaly', methods=['POST'])


def detect_anomaly():

    if request.method == 'POST':
        # Get the uploaded file from the request
        video_file = request.files['file']

        VD = [file for file in os.listdir('Dataset/Human Activity Recognition - Video Dataset') if not file.startswith('.')]
        HMDB = [file for file in os.listdir('Dataset/HMDB_dataset') if not file.startswith('.')]
        NF = [file for file in os.listdir('Dataset/Peliculas') if not file.startswith('.')]
        CLASSES_LIST = VD+NF+HMDB

        # Save the uploaded video to a temporary location
        video_path = 'static/uploaded_video.mp4'
        video_file.save(video_path)

        # Extract frames from the uploaded video
        frames = frames_extraction(video_path)
        frames = np.asarray(frames)

        # Predict using your trained model
        prediction = model.predict(np.expand_dims(frames, axis=0))

        # Get the class index with the highest probability
        predicted_class_index = np.argmax(prediction)

        # Map the class index to the class name
        predicted_class_name = CLASSES_LIST[predicted_class_index]

        print(predicted_class_name)

        return render_template('result.html', predicted_class=predicted_class_name)


def frames_extraction(video_path):
    # Declare a list to store video frames.
    # Specify the height and width to which each video frame will be resized in our dataset.
    IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

    # Specify the number of frames of a video that will be fed to the model as one sequence.
    SEQUENCE_LENGTH = 30
    frames_list = []
    
    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video. 
        success, frame = video_reader.read() 

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        
        # Append the normalized frame into the frames list
        frames_list.append(normalized_frame)
    
    # Release the VideoCapture object. 
    video_reader.release()

    # Return the frames list.
    return frames_list

if __name__ == '__main__':
    app.run(debug=True)