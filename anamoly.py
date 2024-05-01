import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras;
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Dropout

# seed_constant = 5
# np.random.seed(seed_constant)
# random.seed(seed_constant)
# tf.random.set_seed(seed_constant)

# DB_NAMES = ['Human Activity Recognition - Video Dataset', 'HMDB_dataset', 'Peliculas']

# VD = [file for file in os.listdir('Dataset/Human Activity Recognition - Video Dataset') if not file.startswith('.')]
# HMDB = [file for file in os.listdir('Dataset/HMDB_dataset') if not file.startswith('.')]
# NF = [file for file in os.listdir('Dataset/Peliculas') if not file.startswith('.')]
# allDB = VD+NF+HMDB
# print(allDB)

# plt.figure(figsize = (20, 20))

# all_classes_names = allDB
# print(all_classes_names)

# for counter, random_index in enumerate(range(len(all_classes_names)), 1):
#     selected_class_Name = all_classes_names[random_index]

#     # DB Name get
#     for item in VD:
#         if selected_class_Name == item:
#             db_Name = 'Human Activity Recognition - Video Dataset'

#     for item in HMDB:
#         if selected_class_Name == item:
#             db_Name = 'HMDB_dataset'

#     for item in NF:
#         if selected_class_Name == item:
#             db_Name = 'Peliculas'

#     # print(selected_class_Name +" "+db_Name)
            
#     video_files_names_list = [file for file in os.listdir(f'Dataset/{db_Name}/{selected_class_Name}') if not file.startswith('.')]

#     selected_video_file_name = random.choice(video_files_names_list)
 
#     video_reader = cv2.VideoCapture(f'Dataset/{db_Name}/{selected_class_Name}/{selected_video_file_name}')
#     video_reader.set(1, 25)

#     _, bgr_frame = video_reader.read()  
#     bgr_frame = cv2.resize(bgr_frame ,(224,224))

#     video_reader.release()
 
#     rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) 

#     cv2.putText(rgb_frame, selected_class_Name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    
#     # plt.subplot(5, 4, counter);plt.imshow(rgb_frame);plt.axis('off')

# # plt.show()

# # Specify the height and width to which each video frame will be resized in our dataset.
# IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
 
# # Specify the number of frames of a video that will be fed to the model as one sequence.
# SEQUENCE_LENGTH = 30
 
# # Specify the directory containing the UCF50 dataset. DATASET_DIR = "Dataset/Peliculas"
# CLASSES_LIST = all_classes_names


# def frames_extraction(video_path):
#     '''
#     This function will extract the required frames from a video after resizing and normalizing them.
#     Args:
#         video_path: The path of the video in the disk, whose frames are to be extracted.
#     Returns:
#         frames_list: A list containing the resized and normalized frames of the video.
#     '''

#     # Declare a list to store video frames.
#     frames_list = []
    
#     # Read the Video File using the VideoCapture object.
#     video_reader = cv2.VideoCapture(video_path)

#     # Get the total number of frames in the video.
#     video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Calculate the the interval after which frames will be added to the list.
#     skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

#     # Iterate through the Video Frames.
#     for frame_counter in range(SEQUENCE_LENGTH):

#         # Set the current frame position of the video.
#         video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

#         # Reading the frame from the video. 
#         success, frame = video_reader.read() 

#         # Check if Video frame is not successfully read then break the loop
#         if not success:
#             break

#         # Resize the Frame to fixed height and width.
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
#         # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
#         normalized_frame = resized_frame / 255
        
#         # Append the normalized frame into the frames list
#         frames_list.append(normalized_frame)
    
#     # Release the VideoCapture object. 
#     video_reader.release()

#     # Return the frames list.
#     return frames_list

# def create_dataset():
#     '''
#     This function will extract the data of the selected classes and create the required dataset.
#     Returns:
#         features:          A list containing the extracted frames of the videos.
#         labels:            A list containing the indexes of the classes associated with the videos.
#         video_files_paths: A list containing the paths of the videos in the disk.
#     '''

#     # Declared Empty Lists to store the features, labels and video file path values.
#     features = []
#     labels = []
#     video_files_paths = []
    
#     # Iterating through all the classes mentioned in the classes list
#     for class_index, class_name in enumerate(CLASSES_LIST):
        
#         # Display the name of the class whose data is being extracted.
#         print(f'Extracting Data of Class: {class_name}')

#         # ------------------------
#         # DB Name get
#         for item in VD:
#             if class_name == item:
#                 db_Name = 'Human Activity Recognition - Video Dataset'

#         for item in HMDB:
#             if class_name == item:
#                 db_Name = 'HMDB_dataset'

#         for item in NF:
#             if class_name == item:
#                 db_Name = 'Peliculas'
#         # ------------------------
        
#         DATASET_DIR = f'Dataset/{db_Name}'
        
#         # Get the list of video files present in the specific class name directory.
#         files_list = [file for file in os.listdir(os.path.join(DATASET_DIR, class_name)) if not file.startswith('.')]
#         # os.listdir(os.path.join(DATASET_DIR, class_name))
        
        
#         # Iterate through all the files present in the files list.
#         for file_name in files_list:
            
#             # Get the complete video path.
#             video_file_path = os.path.join(DATASET_DIR, class_name, file_name)

#             # Extract the frames of the video file.
#             frames = frames_extraction(video_file_path)

#             # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
#             # So ignore the vides having frames less than the SEQUENCE_LENGTH.
#             if len(frames) == SEQUENCE_LENGTH:

#                 # Append the data to their repective lists.
#                 features.append(frames)
#                 labels.append(class_index)
#                 video_files_paths.append(video_file_path)

#     # Converting the list to numpy arrays
#     features = np.asarray(features)
#     labels = np.array(labels)  
    
#     # Return the frames, class index, and video file path.
#     return features, labels, video_files_paths

# # Create the dataset.
# features, labels, video_files_paths = create_dataset()


# # print(features)

# # Using Keras's to_categorical method to convert labels into one-hot-encoded vectors
# one_hot_encoded_labels = to_categorical(labels)

# # Split the Data into Train ( 75% ) and Test Set ( 25% ).
# features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.25, shuffle = True, random_state = seed_constant)
# features = None
# labels = None


# def create_LRCN_model():
#     '''
#     This function will construct the required LRCN model.
#     Returns:
#         model: It is the required constructed LRCN model.
#     '''

#     # We will use a Sequential model for model construction.
#     model = Sequential()
    
#     # Define the Model Architecture.
#     ########################################################################################################################
    
#     model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu'), input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
#     model.add(TimeDistributed(MaxPooling2D((4, 4))))
    
#     model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
#     model.add(TimeDistributed(MaxPooling2D((4, 4))))
    
#     model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same',activation = 'relu')))
#     model.add(TimeDistributed(MaxPooling2D((2, 2))))
    
#     model.add(TimeDistributed(Conv2D(256, (2, 2), padding='same',activation = 'relu')))
#     model.add(TimeDistributed(MaxPooling2D((2, 2))))
                                      
#     model.add(TimeDistributed(Flatten()))
                                      
#     model.add(LSTM(32))
                                      
#     model.add(Dense(len(CLASSES_LIST), activation = 'softmax'))

#     ########################################################################################################################

#     # Display the models summary.
#     model.summary()
    
#     # Return the constructed LRCN model.
#     return model

# model = create_LRCN_model()

# early_stopping_callback = EarlyStopping(monitor = 'val_accuracy', patience = 10, mode = 'max', restore_best_weights = True)

# # Compile the model and specify loss function, optimizer and metrics to the model.
# model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics = ["accuracy"])
 
# # Start training the model.
# model_training_history = model.fit(x = features_train, y = labels_train, epochs = 70, batch_size = 4 , shuffle = True, validation_split = 0.25, callbacks = [early_stopping_callback])
# model.save("Suspicious_Human_Activity_Detection_LRCN_Model.h5")


# def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
#     # Get metric values using metric names as identifiers.
#     metric_value_1 = model_training_history.history[metric_name_1]
#     metric_value_2 = model_training_history.history[metric_name_2]
    
#     # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
#     epochs = range(len(metric_value_1))
 
#     # Plot the Graph.
#     plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
#     plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
 
#     # Add title to the plot.
#     plt.title(str(plot_name))
#     plt.legend()



# # plot_metric(model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
# plot_metric(model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')
# plt.show()

# # Calculate Accuracy On Test Dataset
# acc = 0
# for i in range(len(features_test)):
#   predicted_label = np.argmax(model.predict(np.expand_dims(features_test[i],axis =0))[0])
#   actual_label = np.argmax(labels_test[i])
#   if predicted_label == actual_label:
#       acc += 1
# acc = (acc * 100)/len(labels_test)
# print("Accuracy =",acc)
# # Accuracy = 42.857142857142854