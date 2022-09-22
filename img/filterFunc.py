from tkinter import *
import os
from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils

facial_features_cordinates = {}
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
])
def shape_to_numpy_array(shape, dtype="int"):
    

    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    

    # return the list of (x, y)-coordinates
    return coordinates


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.2):

    

    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    

    # if the colors list is None, initialize it with a unique

    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        

        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]

        

        pts = shape[j:k]
        facial_features_cordinates[name] = pts
		

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        
        hull = cv2.convexHull(pts)
        cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    # print(facial_features_cordinates)
    return output


path_dir = os.path.dirname(os.path.realpath(__file__))
file_list = os.listdir(path_dir)
real_file_list = [x for x in file_list if(x.endswith(".PNG") or (x.endswith(".png")==True))]
print(real_file_list)

xn=0
root=Tk()
root.title("LipFilter")
root.geometry("720x720")
root.resizable(0, 0)
image=PhotoImage(file=path_dir+"/"+real_file_list[xn])
editimage = PhotoImage(file = "D:\openCV\ethanKim.png")
editimage = editimage.zoom(25) #with 250, I ended up running out of memory
editimage = editimage.subsample(32)
def showimg_p():
	global xn
	global image
	xn-=1
	if(xn>=len(real_file_list)):
		xn=0
	image=PhotoImage(file=path_dir+"/"+real_file_list[xn])
	label_2 = Label(root, image=image)
	label_2.place(x=110,y=60)
def showimg_n():
	global xn
	global image
	xn+=1
	if(xn>=len(real_file_list)):
		xn=0
	if(xn<0):
		xn = len(real_file_list)-1
	image=PhotoImage(file=path_dir+"/"+real_file_list[xn])
	label_2 = Label(root, image=image)
	label_2.place(x=110,y=60)

# def showimg_change():
# 	global editimage
# 	editimage=PhotoImage(file="D:\openCV\sample.png")
# 	label_3 = Label(root, image=editimage)
# 	label_3.place(x=110,y=160)

def switch(key):
	colors = {0 : [(0, 0, 139)]
	 , 1 : [(42, 42, 165)], 2: [(45, 82, 160)], 3: [(19, 69, 139)]
	 , 4: [(92, 92, 205)], 5: [(143, 143, 188)], 6: [(128, 128, 240)]
	 , 7: [(114, 128, 250)], 8: [(122, 150, 233)], 9: [(80, 127, 255)]
	 , 10: [(71, 99, 255)], 11: [(96, 164, 244)], 12: [(122, 160, 255)]}.get(key, [(0, 0, 139)]) #default : darkred
	return colors

def showimg_change1():
	global editimage
	global xn
	colorL = switch(xn)
	



	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('D:\openCV\cv_env\Lib\site-packages\shape_predictor_68_face_landmarks.dat')
	faceimage=cv2.imread('ethanKim.jpg')
	faceimage = imutils.resize(faceimage, width=550)
	gray = cv2.cvtColor(faceimage, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = shape_to_numpy_array(shape)
		output = visualize_facial_landmarks(faceimage, shape,colors = colorL)
		cv2.imwrite("D:\openCV\sample.png",output)
	editimage=PhotoImage(file="D:\openCV\sample.png")
	label_3 = Label(root, image=editimage)
	label_3.place(x=110,y=160)	

btn = Button(root,text="next",command=showimg_n,width=7,height=1) 
btn2 = Button(root,text="previous",command=showimg_p,width=7,height=1) 
okbtn = Button(root, text = "change", command = showimg_change1, width= 7, height=1)
label_1 = Label(root,text="Choose the color!",font="NanumGothic 20")
label_2 = Label(root, image=image)
label_1.place(x=200,y=10)
label_2.place(x=110,y=60)
btn.place(x=450,y=100)
btn2.place(x=150,y=100)
okbtn.place(x=300, y =100)


label_3 = Label(root, image=editimage)
label_3.place(x=110,y=160)


root.mainloop()