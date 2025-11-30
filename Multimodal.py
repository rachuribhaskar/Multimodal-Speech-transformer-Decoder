from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle


main = tkinter.Tk()
main.title("Multi-modal Speech Transformer Decoders: When Do Multiple Modalities Improve Accuracy") 
main.geometry("1300x1200")

global filename, transformer_model, audio, images, Y
global images_train, images_test, audio_train, audio_test, Y_train, Y_test
global labels, scaler, dataset

def getLabel(name):
    arr = name.split("/")
    index = -1
    for i in range(len(labels)):
        if labels[i] == arr[1]:
            index = i
            break        
    return index

def upload(): 
    global filename, labels, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")
    labels = ['beach', 'classroom1', 'forest', 'jungle', 'london']
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset)+"\n\n")
    text.update_idletasks()   

def preprocess():
    text.delete('1.0', END)
    global dataset,  audio, images, Y, scaler
    if os.path.exists('model/images.npy'):
        images = np.load('model/images.npy')
        Y = np.load('model/Y.npy')
        audio = np.load("model/audio.npy")
    else:
        images = []
        audio = []
        Y = []
        dataset = dataset.values
        for i in range(len(dataset)):
            mfcc = dataset[i].ravel()
            label = getLabel(mfcc[0])
            img = cv2.imread("Dataset/"+mfcc[0])
            img = cv2.resize(img, (32, 32))
            images.append(img)
            mfcc = mfcc[1:len(mfcc)-2]
            mfcc = np.asarray(mfcc)
            mfcc = mfcc.astype(float)
            audio.append(mfcc)
            Y.append(label)
            print(str(label)+" "+str(mfcc.shape))
        images = np.asarray(images)
        audio = np.asarray(audio)
        Y = np.asarray(Y)
        np.save('model/images',images)
        np.save('model/Y',Y)
        np.save("model/audio", audio)
    images = images.astype('float32')
    images = images/255
    scaler = StandardScaler()
    audio = scaler.fit_transform(audio)
    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)
    images = images[indices]
    Y = Y[indices]
    audio = audio[indices]
    Y = to_categorical(Y)
    images = np.reshape(images, (images.shape[0], (images.shape[1] * images.shape[2] * images.shape[3])))
    text.insert(END,"Dataset Images & Audio Processing Completed\n")
    text.insert(END,"Total images and audio files found in dataset = "+str(images.shape[0])+"\n")
    text.insert(END,"Total features extracted & processed from images = "+str(images.shape[1])+"\n")
    text.insert(END,"Total features extracted & processed from audio = "+str(audio.shape[1])+"\n")

def trainTestSplit():
    text.delete('1.0', END)
    global images, Y, audio
    global images_train, images_test, audio_train, audio_test, Y_train, Y_test
    images_train, images_test, audio_train, audio_test, Y_train, Y_test = train_test_split(images, audio, Y, test_size=0.2, random_state=42)
    text.insert(END,"Dataset split for train and test\n\n")
    text.insert(END,"80% dataset records used to train Transformer Decoder Model : "+str(images_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to test Transformer Decoder Model : "+str(images_test.shape[0])+"\n")    

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")

def transformerLayer(query, value):
    transformer = layers.Dot(axes=(2, 2))([query, value])
    transformer = layers.Activation('softmax')(transformer)
    context = layers.Dot(axes=(2, 1))([transformer, value])
    return context

def build_encoder(audio_embedding_dim, image_feature_dim):
    audio_input = layers.Input(shape=(audio_embedding_dim,))
    image_features_input = layers.Input(shape=(image_feature_dim,))
    audio_processed = layers.Dense(128, activation='relu')(audio_input)
    audio_processed = layers.Reshape((1, 128))(audio_processed) #reshape to perform dot product.
    image_features_processed = layers.Dense(128, activation='relu')(image_features_input)
    image_features_processed = layers.Reshape((1, 128))(image_features_processed) #reshape to perform dot product.
    # applying transformer
    transformer_features = transformerLayer(audio_processed, image_features_processed)
    transformer_features = layers.Reshape((128,)) (transformer_features)
    # Combine transformer features with other encoder layers
    combined_features = layers.Concatenate()([transformer_features, image_features_input])
    # defining decoder layer
    x = layers.Dense(256, activation='relu')(combined_features)
    x = layers.Dense(5, activation='softmax')(x)  # Example: Flattened image output
    output_image = (x) #reshape to image dimensions.
    model = keras.Model(inputs=[audio_input, image_features_input], outputs=output_image)
    return model

def trainTransformer():
    text.delete('1.0', END)
    global images_train, images_test, audio_train, audio_test, Y_train, Y_test, transformer_model
    audio_embedding_dim = 104
    image_feature_dim = 3072
    transformer_model = build_encoder(audio_embedding_dim, image_feature_dim)
    transformer_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/transformer_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/transformer_weights.hdf5', verbose = 1, save_best_only = True)
        hist = transformer_model.fit([audio_train, images_train], Y_train, batch_size = 32, epochs = 30, validation_data=([audio_test, images_test], Y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/transformer_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        transformer_model.load_weights("model/transformer_weights.hdf5")
    #perform prediction on test data    
    predict = transformer_model.predict([audio_test, images_test])
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(Y_test, axis=1)
    calculateMetrics("Transformer Multi-modal", y_test1, predict)

def graph():
    f = open('model/transformer_history.pckl', 'rb')
    train_values = pickle.load(f)
    f.close()
    print(train_values)
    accuracy_value = train_values["val_acc"]
    loss_value = train_values["val_loss"]
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Loss')
    plt.plot(accuracy_value, 'ro-', color = 'green')
    plt.plot(loss_value, 'ro-', color = 'blue')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    plt.title('Transformer Training Accuracy & Loss Graph')
    plt.show()

def predict():
    text.delete('1.0', END)
    global transformer_model, scaler, labels
    filename = filedialog.askdirectory(initialdir="testData")
    testdata = pd.read_csv(filename+"/audio.csv")
    testdata = testdata.values
    mfcc = testdata[0]
    mfcc = np.asarray(mfcc)
    mfcc = mfcc.astype(float)
    test = []
    test.append(mfcc)
    mfcc = scaler.transform(test)
    img = cv2.imread(filename+"/image.png")
    img = cv2.resize(img, (32, 32))
    img = img.reshape(1,32,32,3)
    img = img.astype('float32')
    img = img/255
    img = np.reshape(img, (img.shape[0], (img.shape[1] * img.shape[2] * img.shape[3])))
    predict = transformer_model.predict([mfcc, img])
    predict = np.argmax(predict)
    predict = labels[predict]
    img = cv2.imread(filename+"/image.png")
    img = cv2.resize(img, (600,400))#display image with predicted output
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.putText(img, 'Speech & Image Features Recognized as : '+predict, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    plt.imshow(img)
    plt.title('Speech & Image Features Recognized as : '+predict)
    plt.show()
            
def close():
    main.destroy()


font = ('times', 16, 'bold')
title = Label(main, text='Multi-modal Speech Transformer Decoders: When Do Multiple Modalities Improve Accuracy')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=24,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=40,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Audio & Images Dataset", command=upload, bg='#ffb3fe')
uploadButton.place(x=50,y=600)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocess, bg='#ffb3fe')
processButton.place(x=350,y=600)
processButton.config(font=font1) 

splitButton1 = Button(main, text="Train & Test Split", command=trainTestSplit, bg='#ffb3fe')
splitButton1.place(x=560,y=600)
splitButton1.config(font=font1) 

lstmButton = Button(main, text="Train Multi-modal Transformer", command=trainTransformer, bg='#ffb3fe')
lstmButton.place(x=780,y=600)
lstmButton.config(font=font1) 

tcnButton = Button(main, text="Training Graph", command=graph, bg='#ffb3fe')
tcnButton.place(x=50,y=650)
tcnButton.config(font=font1) 

graphButton = Button(main, text="Speech Recognition from Audio & Image", command=predict, bg='#ffb3fe')
graphButton.place(x=240,y=650)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close, bg='#ffb3fe')
exitButton.place(x=580,y=650)
exitButton.config(font=font1)

main.config(bg='LightSalmon3')
main.mainloop()
