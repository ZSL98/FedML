import os
import sys

os.chdir(sys.path[0])
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
print(sys.path)
from fedml_api.data_preprocessing.MNIST.data_loader import read_data

train_path="./train"
test_path="./test"

users, groups, train_data, test_data = read_data(train_path, test_path)

print(users)