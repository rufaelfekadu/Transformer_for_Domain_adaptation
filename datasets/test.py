import random

# Define the paths of the input and output files
input_file_path = "/home/rufael.marew/Documents/Academics/AI702/project/test/sgada_data/flir/flir.txt"
train_output_file_path = "../data/train_labels.txt"
val_output_file_path = "../data/val_labels.txt"

# Define the percentage of data to use for training (the rest will be used for validation)
train_percent = 0.8

# Read the labels from the input file
with open(input_file_path, "r") as input_file:
    labels = input_file.readlines()

# Shuffle the labels randomly
random.shuffle(labels)

# Determine the number of labels to use for training and validation
num_train = int(len(labels) * train_percent)
num_val = len(labels) - num_train

# Write the training labels to the output file
with open(train_output_file_path, "w") as train_output_file:
    train_output_file.writelines(labels[:num_train])

# Write the validation labels to the output file
with open(val_output_file_path, "w") as val_output_file:
    val_output_file.writelines(labels[num_train:])
