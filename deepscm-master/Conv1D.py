# Import libraries
import numpy as np

# Import machine learning libraries
import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Import custom functions
from utils import one_hot_encoder, create_conv1D

def Conv1D_regression(X, y, params=None):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    X_train = [one_hot_encoder(s=x) for x in X_train]
    X_train = np.transpose(np.asarray(X_train), (0, 2, 1))
    X_train = np.asarray(X_train)

    X_test = [one_hot_encoder(s=x) for x in X_test]
    X_test = np.transpose(np.asarray(X_test), (0, 2, 1))
    X_test = np.asarray(X_test)
    
    X_val = [one_hot_encoder(s=x) for x in X_val]
    X_val = np.transpose(np.asarray(X_val), (0, 2, 1))
    X_val = np.asarray(X_val)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_val = np.asarray(y_val)

    Conv1D_regression = create_conv1D((272, 21))

    # Compile the Conv1D
    Conv1D_regression.compile(
        optimizer='adam', loss='mae', metrics=None
    )

    # Create callback
    filepath = 'Conv1D_regression.h5'
    checkpoint = ModelCheckpoint(filepath=filepath, 
                                 monitor='val_loss',
                                 verbose=1, 
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min')
    callbacks = [checkpoint] 

    # Fit the CNN to the training set
    history = Conv1D_regression.fit(
       x=X_train, y=y_train, shuffle=True, validation_data=(X_val, y_val),
        epochs=50, callbacks=callbacks, batch_size=64, verbose=2
    )

    # Save the Conv1D architecture to json
    Conv1D_regression_json = Conv1D_regression.to_json()
    with open("Conv1D_regression.json", "w") as json_file:
            json_file.write(Conv1D_regression_json)

    # Load the Conv1D architecture from json
    pred_model = model_from_json(Conv1D_regression_json)
 
    # Load weights from the best model into Conv1D model
    pred_model.load_weights("Conv1D_regression.h5")

    # Compile the loaded Conv1D model
    pred_model.compile(optimizer='adam', metrics=['mae'])

    # Evaluate the performace
    score = pred_model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f" % (pred_model.metrics_names[1], score[1]))

    return 1 
