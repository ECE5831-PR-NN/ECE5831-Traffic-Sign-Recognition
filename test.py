import numpy as np
import cv2
print ('hello world')




# Load an color image in grayscale
##img = cv2.imread('C:\Users\ATAYL232\Pictures\Random\Juneau\20180628_124929.jpg')
##cv2.imshow('image',img)


  
# path 
#path = r'C:\Users\ATAYL232\Pictures\Screenshots\Screenshot (1).png'
path = r'C:\Users\Splinter\Pictures\GTSRB\Final_Training\Images\00001\00000_00001.ppm'
  
# Using cv2.imread() method 
img = cv2.imread(path) 

#split into red, blue green
b,g,r = cv2.split(img)

edges = cv2.Canny(img,100,200)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(b)
print(g)
print(r)
# Displaying the image 
cv2.imshow('originalimage', img) 
cv2.imshow('red', r)
cv2.imshow('Edge Image', edges)
cv2.imshow('Gray',gray)
cv2.waitKey(0)
#cv2.destroyAllWindows()