import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Define the paths of the four models
model_paths = [
    '/home/ajay/Desktop/riya/result_of_5_fold/DenseNet121/DenseNet_model_fold4.h5',
    '/home/ajay/Desktop/riya/result_of_5_fold/MobileNetv2/MobileNet_model_fold2.h5',
    '/home/ajay/Desktop/riya/result_of_5_fold/ResNet50/ResNet_model_fold1.h5',
    '/home/ajay/Desktop/riya/result_cnn/saved_models/VGG_model.h5'
]

# Load the saved models
models = [tf.keras.models.load_model(path) for path in model_paths]

classes = ['Adenocarcinoma', 'Benign', 'Squamous cell carcinoma']

# Function to classify an image
def classify_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize image to match model input size
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        selected_model = models[model_selection.current()]
        prediction = selected_model.predict(img)
        predicted_class = classes[np.argmax(prediction)]
        confidence_score = prediction[0][np.argmax(prediction)]
        # messagebox.showinfo("Prediction", f"The predicted subtype of lung cancer is : {predicted_class}\n\n Confidence Score :{confidence_score:.2f}")
        output_display.config(text=f"The predicted subtype of \nlung cancer is : \n{predicted_class}\n\n Confidence Score :{confidence_score:.2f}",font=("Times New Roman",11,"bold"))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to open file dialog and classify selected image
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        classify_image(file_path)

# Function to display additional information
def display_info():
    messagebox.showinfo("Information", '''This is multi-class classifier which classifies among benign, squamous cell carcinoma and adenocarcinoma.
    Benign- do not invade nearby tissue or invade other parts of the body ,often asymptomatics and rare in occurrence
    Malignant- spreads to other body parts, majorly have two subtypes -Adenocarcinoma(LUAD) and squamous cell carcinoma(LUSC)
    LUAD- mostly found in outer regions of the lungs, most prevalent type, centrally located, mucin production occurs
    LUSC- mostly occures in smokers,associated with hypercalcimia, survival chances are better than LUAD ''')
def display_ref():
    messagebox.showinfo("References", "GUI build using: \n Tkinter 0.1.0 \n Tensorflow 2.13.1  \n Numpy 1.24.3  \n Pillow 10.3.0 \n Kaggle.com ")    

# Function to switch between frames
def switch_frame(frame):
    try:
        frame.tkraise()
    except Exception as e:
        print(f"Error switching frames: {e}")


# Function to close the application
def close_frame2():
    # frame_two.deletecommand()
    switch_frame(frame_one)

# Function to classify image and switch to Frame Two
def classify_and_switch():
    open_file_dialog()
    switch_frame(frame_two)


# Create the main application window
root = tk.Tk()
root.title("Classification Model")
root.geometry("800x600")
root.minsize(300, 200)
root.maxsize(1600, 1200)

bg_image = tk.PhotoImage(file="/home/ajay/Desktop/riya/bg5.png")
img_ = tk.PhotoImage(file="/home/ajay/Desktop/riya/lung__.png")
des_img = tk.PhotoImage(file = "/home/ajay/Desktop/riya/description.png")
background_label = tk.Label(root, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

label = tk.Label(root, text="CLASSIFICATION OF HISTOPATHOLOGICAL IMAGES OF \n LUNG CANCER USING CONVOLUTIONAL NEURAL NETWORK", font=("Times New Roman", 22, "bold"),
                 width=80, height=3, bg="burlywood", fg="black",anchor="center")
label.pack(pady=50)

# Frame One
frame_one = tk.Frame(root)
frame_one = tk.Frame(root, width=1000, height=800, bg="lightgray", bd=1, relief=tk.SOLID)
frame_one.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

image_frame = tk.Frame(frame_one, width=400, height=300)
image_frame.pack(side=tk.LEFT, padx=10, pady=10)
image_label = tk.Label(image_frame, image=img_)
image_label.image = img_
image_label.pack(side=tk.LEFT, padx=10, pady=10)


description_frame = tk.Frame(frame_one, width=400, height=400)
description_frame.pack(side=tk.RIGHT, padx=10, pady=10)
des_img_label = tk.Label(description_frame, image=des_img)
des_img_label.image = des_img
des_img_label.pack(side=tk.RIGHT, padx=10, pady=10)


info_button_one = tk.Button(frame_one, height=1,width=2,text="info",font=("Arial",8,"bold"),border=1, command=display_info)
info_button_one.pack(side=tk.BOTTOM)

ref_button_one = tk.Button(frame_one, height=1,width=7,text= "references",font= ("Arial",8,"bold"),border=1 ,command=display_ref)
ref_button_one.pack(side=tk.BOTTOM)

# classify_button = tk.Button(frame_one, text="Classify Image", command=lambda: switch_frame(frame_two))
# classify_button.pack(side=tk.BOTTOM, pady=10)
# Frame One
classify_button = tk.Button(frame_one, text="Classify Image", command=classify_and_switch)
classify_button.pack(side=tk.BOTTOM, pady=10)

# Frame Two
frame_two = tk.Frame(root)
frame_two = tk.Frame(root, bg="lightgray", bd=2, relief=tk.SOLID)
frame_two.place(height=340,width=500,relx=0.5,rely=0.5,anchor=tk.CENTER)

model_selection = ttk.Combobox(frame_two, values=["DenseNet121", "MobileNetv2", "ResNet50", "VGG16"])
model_selection.current(0)
model_selection.pack(padx=2,pady=10)

browse_button = tk.Button(frame_two, text="Browse Image", command=open_file_dialog)
browse_button.pack(pady=10)

img_label= tk.Label(frame_two)
img_label.pack(side=tk.LEFT, padx=10)

output_display = tk.Label(frame_two,height=8, width=30, justify="center",bg="floral white")
output_display.pack(side=tk.RIGHT, padx=(0,12),pady=10)

# Exit button
exit_button = tk.Button(frame_two, text="X", command=close_frame2, bg="red", fg="white", padx=3, pady=2)
exit_button.place(relx=1, rely=0, anchor=tk.NE)

# Initially show frame one
frame_one.tkraise()

root.mainloop()


