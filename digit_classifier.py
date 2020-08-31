import cv2
import numpy as np
import PIL
from PIL import ImageTk, Image, ImageDraw
import tkinter
from tkinter import *
from tensorflow.keras.models import load_model

classes=[0,1,2,3,4,5,6,7,8,9]
width = 500
height = 450

model=load_model('mnist_digit.h5')

def testing():
    img=cv2.imread('image.png',0)
    img=cv2.bitwise_not(img)
    img=cv2.resize(img,(28,28))
    img=img.reshape(1,28,28,1)
    img=img.astype('float32')
    img=img/255.0
    pred=model.predict(img)
    return pred

def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=10)
    draw.line([x1, y1, x2, y2],fill="black",width=10)

def model_pred():
    filename = "image.png"
    image1.save(filename)
    pred=testing()
    txt.insert(INSERT,"{}\nAccuracy: {}%\n".format(classes[np.argmax(pred[0])],round(pred[0][np.argmax(pred[0])]*100,3)))    

def clear():
    cv.delete('all')
    draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
    txt.delete('1.0', END)


win = Tk()

win.resizable(0,0)
cv = Canvas(win, width=width, height=height, bg='white')
cv.pack()

image1 = PIL.Image.new("RGB", (width, height), (255,255,255))
draw = ImageDraw.Draw(image1)

txt= Text(win,bd=3,exportselection=0,bg='white',font='Helvetica', padx=10,pady=10,height=3,width=30)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)


btnModel=Button(text="Predict",command=model_pred,padx=5,pady = 5,width = 20)
btnModel.pack()

btnClear=Button(text="clear",command=clear,padx=5,pady = 5,width = 20)
btnClear.pack()

txt.pack()

win.title('Handwritten Digit Recognition')
win.mainloop()



