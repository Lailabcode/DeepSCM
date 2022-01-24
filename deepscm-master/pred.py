# Import libraries
import numpy as np
import argparse

from keras.models import model_from_json

from Conv1D import Conv1D_regression

# Import custom functions
from utils import one_hot_encoder, load_pred_data

parser = argparse.ArgumentParser(prog='pred.py')
parser.add_argument("--infile", help="DeepSCM input file", type=str, default='DeepSCM_input.txt')
args = parser.parse_args()

name_list, seq_list = load_pred_data(args.infile)

json_file = open('Conv1D_regression.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into model
loaded_model.load_weights("Conv1D_regression.h5")

X = [one_hot_encoder(s=x) for x in seq_list]
X = np.transpose(np.asarray(X), (0, 2, 1))
X = np.asarray(X)

loaded_model.compile(optimizer='adam', loss='mae', metrics=[None])
y_pred = loaded_model.predict(X)

for i in range(len(y_pred)):
    print("%s %.2f" % (name_list[i], y_pred[i]))

