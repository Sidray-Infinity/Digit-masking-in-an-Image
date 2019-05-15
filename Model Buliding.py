

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import tensorflow as tf
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import time
get_ipython().run_line_magic('matplotlib', 'inline')




def makeModel():
    
    train_dir = r'D:\Project\Digit Masking in an Image\Synthetic_dataset\Train'
    test_dir = r'D:\Project\Digit Masking in an Image\Synthetic_dataSet\Test'
    valid_dir = r'D:\Project\Digit Masking in an Image\Synthetic_dataSet\Validation'
    
    t1 = time.time()
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(28, 28),
        batch_size=32,
        color_mode="grayscale",
        class_mode="categorical")
    
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(28, 28),
        batch_size=32,
        color_mode="grayscale",
        class_mode="categorical")
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        batch_size=1,
        color_mode="grayscale",
        shuffle=False,
        class_mode=None)
    
    print('BOY!, It took',time.time() - t1)
    
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    
#     for data_batch, labels_batch in train_generator:
#         print('Train data batch shape:', data_batch.shape)
#         print('Train labels batch shape:', labels_batch.shape)
#         break
        
#     for data_batch, labels_batch in test_generator:
#         print('Test data batch shape:', data_batch.shape)
#         print('Test labels batch shape:', labels_batch.shape)
#         print(labels_batch)
#         break
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(36, activation='softmax'))
    
    model.compile(optimizer=optimizers.RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  epochs=5,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID)
        
    eval_results = model.evaluate_generator(generator=valid_generator,
                             steps=STEP_SIZE_VALID)

    model.save('synthetic_dataset4.h5')
    
    
    test_generator.reset()
    pred = model.predict_generator(test_generator,
                                 steps=STEP_SIZE_TEST,
                                 verbose=1)

    predicted_class_indices=[]
    for i in pred:
        predicted_class_indices.append(np.argmax(i))
    
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames=test_generator.filenames
    results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
    results.to_csv("results.csv",index=False)
    
    print('EVALUATION RESULTS:', eval_results)
    
    return model


model = makeModel()
model.summary()





