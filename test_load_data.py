from load_data_new import *
import os

current_path = os.path.dirname(os.path.abspath(__file__)) 
file_name = "goblet_book.txt"

file_path = os.path.join(current_path, file_name)

train_data, test_data, vocab= load_data(file_path, 5)

