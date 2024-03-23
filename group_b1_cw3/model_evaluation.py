import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report


## CONSTANTS
SLIDING_WINDOW_LENGTH = 125
SENSOR_CHANNELS = 3

activity_label_dict = {'ascending': 0,
 'descending': 1,
 'lyingBack': 2,
 'lyingLeft': 3,
 'lyingRight': 4,
 'lyingStomach': 5,
 'miscMovement': 6,
 'normalWalking': 7,
 'running': 8,
 'shuffleWalking': 9,
 'sitStand': 10}

sub_activity_label_dict = {'breathingNormal': 0, 'coughing': 1, 'hyperventilating': 2, 'other': 3}

stationary_dict = {'lyingBack':0, 'lyingLeft':1, 'lyingRight':2, 'lyingStomach':3, 'sitStand':4}

stationary = set([2,3,4,5,10])

label_sub_activity_dict = {value: key for key, value in sub_activity_label_dict.items()}

label_stationary_dict = {value: key for key, value in stationary_dict.items()}

## HELPERS

def sliding_window(X, label_col, n_output):
    step = 10
    segments = []
    labels = []
    act_labels, sub_labels = [], []

    for i in range(0, len(X)-SLIDING_WINDOW_LENGTH, step):
        xs = X['accel_x'].values[i: i+SLIDING_WINDOW_LENGTH]
        ys = X['accel_y'].values[i: i+SLIDING_WINDOW_LENGTH]
        zs = X['accel_z'].values[i: i+SLIDING_WINDOW_LENGTH]
        
        if label_col == 'activity_label':
            d = Counter(X[label_col][i: i+SLIDING_WINDOW_LENGTH])

            if len(d)==1:
                label = list(d.keys())[0]
                segments.append([xs, ys, zs])
                labels.append(label)
        else:
            d_act = Counter(X['activity_label'][i: i+SLIDING_WINDOW_LENGTH])
            d_sub = Counter(X['sub_activity_label'][i: i+SLIDING_WINDOW_LENGTH])
            
            if len(d_act)==1 and len(d_sub)==1:
                label_act = list(d_act.keys())[0]
                label_sub = list(d_sub.keys())[0]
                segments.append([xs, ys, zs])
                act_labels.append(label_act)
                sub_labels.append(label_sub)

    reshaped_segments = np.asarray(segments)
    
    if n_output!=11:
        act_segments = reshaped_segments
        reshaped_segments = reshaped_segments.reshape(-1, SLIDING_WINDOW_LENGTH, SENSOR_CHANNELS)
        act_labels = np.asarray(act_labels)
        sub_labels = np.asarray(sub_labels)
        return act_segments, reshaped_segments, act_labels, sub_labels
    
    labels = np.asarray(labels)
    return reshaped_segments, labels


## Get model path and test data path from args
parser = argparse.ArgumentParser(description='A script with command-line arguments.')

parser.add_argument('--model_path', help='Path to tflite models folder', required=True)
parser.add_argument('--test_data_path', help='Path to Test data csv file', required=True)

args = parser.parse_args()

model_path = args.model_path
test_data_path = args.test_data_path


## Read and prepare the test dataset
test_dataset = pd.read_csv(test_data_path)

test_dataset['activity'] = list(map(lambda x: x.split('_')[0], test_dataset['class']))
test_dataset['sub_activity'] = list(map(lambda x: x.split('_')[1], test_dataset['class']))

test_dataset['activity_label'] = test_dataset['activity'].replace(activity_label_dict)
test_dataset['sub_activity_label'] = test_dataset['sub_activity'].replace(sub_activity_label_dict)


## TASK 1

# load task1 model
interpreter = tf.lite.Interpreter(model_path=f'{model_path}/task1.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_tensor_index = input_details[0]['index']

# generate input vectors for Task1
task1_dataset = test_dataset[test_dataset.sub_activity == 'breathingNormal']
Xtst, ytst = sliding_window(task1_dataset, 'activity_label', 11)

# generate evaluation report of model
ypred = []
for i,x in enumerate(Xtst):
    interpreter.set_tensor(input_tensor_index, np.array([x.astype(np.float32)]))
    interpreter.invoke()
    ypred.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))
    
report = classification_report(ytst, ypred, target_names=activity_label_dict.keys())

print('---------------------TASK 1----------------------')
print(report)

## TASK 2

# generate input vectors for Task2
task2_dataset = test_dataset[(test_dataset['activity_label'].isin(stationary)) & (test_dataset.sub_activity != 'other')].copy()
task2_dataset['activity_label'] = task2_dataset['activity'].replace(stationary_dict)
Xtst_act, Xtst_sub, ytst_act, ytst_sub = sliding_window(task2_dataset, 'sub_activity_label', 3)

# load task2 activities model
interpreter = tf.lite.Interpreter(model_path=f'{model_path}/task2-activities.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_tensor_index = input_details[0]['index']

# get predictions from activities model
ypred_act = []
for i,x in enumerate(Xtst_act):
    interpreter.set_tensor(input_tensor_index, np.array([x.astype(np.float32)]))
    interpreter.invoke()
    ypred_act.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))

# load task2 subactivities model
interpreter = tf.lite.Interpreter(model_path=f'{model_path}/task2-sub.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_tensor_index = input_details[0]['index']

# get predictions from subactivities model
ypred_sub = []
for i,x in enumerate(Xtst_sub):
    interpreter.set_tensor(input_tensor_index, np.array([x.astype(np.float32)]))
    interpreter.invoke()
    ypred_sub.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))

# combine activities and subactivities
ytst = list(map(lambda x: f'{label_stationary_dict[x[0]]}_{label_sub_activity_dict[x[1]]}', zip(ytst_act, ytst_sub)))
ypred = list(map(lambda x: f'{label_stationary_dict[x[0]]}_{label_sub_activity_dict[x[1]]}', zip(ypred_act, ypred_sub)))

target_names = np.unique(ytst+ypred)
    
# generate evaluation report of model
report = classification_report(ytst, ypred, target_names=target_names)

print('---------------------TASK 2----------------------')
print(report)

## TASK 3

# generate input vectors for Task3
task3_dataset = test_dataset[(test_dataset['activity_label'].isin(stationary))].copy()
task3_dataset['activity_label'] = task3_dataset['activity'].replace(stationary_dict)
Xtst_act, Xtst_sub, ytst_act, ytst_sub = sliding_window(task3_dataset, 'sub_activity_label', 4)

# load task3 activities model
interpreter = tf.lite.Interpreter(model_path=f'{model_path}/task3-activities.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_tensor_index = input_details[0]['index']

# get predictions from activities model
ypred_act = []
for i,x in enumerate(Xtst_act):
    interpreter.set_tensor(input_tensor_index, np.array([x.astype(np.float32)]))
    interpreter.invoke()
    ypred_act.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))

# load task2 subactivities model
interpreter = tf.lite.Interpreter(model_path=f'{model_path}/task3-sub.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_tensor_index = input_details[0]['index']

# get predictions from subactivities model
ypred_sub = []
for i,x in enumerate(Xtst_sub):
    interpreter.set_tensor(input_tensor_index, np.array([x.astype(np.float32)]))
    interpreter.invoke()
    ypred_sub.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))

# combine activities and subactivities
ytst = list(map(lambda x: f'{label_stationary_dict[x[0]]}_{label_sub_activity_dict[x[1]]}', zip(ytst_act, ytst_sub)))
ypred = list(map(lambda x: f'{label_stationary_dict[x[0]]}_{label_sub_activity_dict[x[1]]}', zip(ypred_act, ypred_sub)))

target_names = np.unique(ytst+ypred)

# generate evaluation report of model
report = classification_report(ytst, ypred, target_names=target_names)

print('---------------------TASK 3----------------------')
print(report)