# Import necessary libraries
import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

# Define the function to generate data paths with labels
def define_paths(dir):
    filepaths = []
    labels = []

    classes = os.listdir(dir)
    for cls in classes:
        cls_path = os.path.join(dir, cls)
        for file in os.listdir(cls_path):
            filepaths.append(os.path.join(cls_path, file))
            labels.append(cls)

    return filepaths, labels

# Define the function to create dataframe from data
def define_df(files, classes):
    Fseries = pd.Series(files, name='filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis=1)

# Define the function to create model data
def create_model_data(train_df, valid_df, batch_size):
    img_size = (224, 224)
    color = 'rgb'

    tr_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    ts_gen = ImageDataGenerator(rescale=1./255)

    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                           target_size=img_size, class_mode='categorical',
                                           color_mode=color, shuffle=True, batch_size=batch_size)

    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                           target_size=img_size, class_mode='categorical',
                                           color_mode=color, shuffle=True, batch_size=batch_size)

    return train_gen, valid_gen

# Define the function to split data into train, valid, test
def split_data(data_dir):
    files, classes = define_paths(data_dir)
    df = define_df(files, classes)
    train_df, test_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123, stratify=classes)
    valid_df, test_df = train_test_split(test_df, train_size=0.5, shuffle=True, random_state=123, stratify=test_df['labels'])
    return train_df, valid_df, test_df

# Define the function to create the model
def create_model(input_shape, num_classes):
    # Load the MobileNetV2 model pretrained on ImageNet without the top layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Combine the base model with custom top layers
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

# Define the directory containing the data
data_dir = '/home/ajay/Desktop/riya/lung_colon_image_set/lung_image_sets'

try:
    # Split data into train, valid, and test
    train_df, valid_df, test_df = split_data(data_dir)

    # Define number of classes
    num_classes = len(train_df['labels'].unique())

    # Get Generators
    batch_size = 32
    img_size=(224,224)
    train_gen, valid_gen = create_model_data(train_df, valid_df, batch_size)

    # Define input shape
    input_shape = (224, 224, 3)  # assuming images are resized to 224x224

    # Define k-fold cross-validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=123)

    fold = 1
    for train_index, val_index in kf.split(train_df):
        print(f"Fold: {fold}")
        train_fold_df = train_df.iloc[train_index]
        val_fold_df = train_df.iloc[val_index]

        # Create generators for this fold
        train_fold_gen, val_fold_gen = create_model_data(train_fold_df, val_fold_df, batch_size)

        # Create and compile the model
        learning_rate = 0.001
        model = create_model(input_shape, num_classes)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(train_fold_gen, epochs=15, validation_data=val_fold_gen)

        #get training and validation accuracy after all epoch 
        tr_ac =history.history['accuracy'][-1]
        val_ac =history.history['val_accuracy'][-1]
        tr_l = history.history['loss'][-1]
        val_l =history.history['val_loss'][-1]
        print(f"Training Accuracy: {tr_ac}, Training loss: {tr_l}")
        print(f"Validation Accuracy: {val_ac}, Validation loss: {val_l}")

        # Save the model
        output_dir = '/home/ajay/Desktop/riya/result_of_5_fold/MobileNetv2'
        model.save(os.path.join(output_dir,f"MobileNet_model_fold{fold}.h5"))

        # Plot training history
        # Define needed variables
        tr_acc = history.history['accuracy']
        tr_loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        index_loss = np.argmin(val_loss)
        val_lowest = val_loss[index_loss]
        index_acc = np.argmax(val_acc)
        acc_highest = val_acc[index_acc]
        Epochs = [i+1 for i in range(len(tr_acc))]
        loss_label = f'best epoch= {str(index_loss + 1)}'  
        acc_label = f'best epoch= {str(index_acc + 1)}'
        
        
        
        # Plot training history
        plt.figure(figsize=(20, 8))
        plt.style.use('fivethirtyeight')

        plt.subplot(1, 2, 1)
        plt.ylim(0, 1) 
        plt.xlim(0,25)
        plt.plot(Epochs, tr_loss, 'r', label='Training loss')
        plt.plot(Epochs, val_loss, 'g', label='Validation loss')
        plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'acc_loss_graph_fold{fold}.png'))  # Save the plot
        
        plt.subplot(1, 2, 2)
        plt.ylim(0,1)
        plt.xlim(0,25)
        plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
        plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
        plt.scatter(index_acc + 1 , acc_highest, s=150, c='blue', label=acc_label)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'acc_loss_graph_fold{fold}.png'))  # Save the plot
        
        plt.tight_layout()
        

        # Evaluate the model on test data
        test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                           target_size=img_size, class_mode='categorical',
                                           color_mode='rgb', shuffle=False, batch_size=batch_size)
        test_loss, test_accuracy = model.evaluate(test_gen)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # Generate predictions
        preds = model.predict(test_gen)
        y_pred = np.argmax(preds, axis=1)
        class_names = list(test_gen.class_indices.keys())

        # Confusion Matrix
        cm = confusion_matrix(test_gen.classes, y_pred)
        plt.figure(figsize=(9, 9))
        sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='g', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, f'cm_fold{fold}.png'))  
        

        # Classification Report
        report = classification_report(test_gen.classes, y_pred, target_names=class_names)
        print("Classification Report:")
        print(report)
        
        fold += 1

except Exception as e:
    print("Error:", e)

