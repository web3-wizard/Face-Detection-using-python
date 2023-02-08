import cv2
from random import randrange

# load some pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

""" For single Image
# choose an image to detect face
img = cv2.imread("spiderman.jpg")
# print(type(img))

# resize the image
img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LINEAR)

# convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(gray_img)
print(face_coordinates)

# draw a rectangle on the image
for t in face_coordinates:
    (x, y, w, h) = t
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)

# show the image
cv2.imshow('Face Detector', img)
cv2.waitKey()
"""

""" For Live video from webcam """
webcam = cv2.VideoCapture(0)
# webcam.open("http://192.168.0.105:8000/")

if not webcam.isOpened:
    print("-- Error opening webcam!")
    exit(0)
while(True):
    # read the current frame
    ret, frame = webcam.read()
    if frame is None:
        print("-- No capture frame -- Break!")
        break

    # convert the frame into grayscale
    gray_scaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces from webcam
    face_coordinates = trained_face_data.detectMultiScale(gray_scaled_frame)

    # draw the rectangle around the faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 5)

    # show the capture video live from webcam
    cv2.imshow('Face Detect From Webcam', frame)

    if cv2.waitKey(10) == 27 or cv2.waitKey(10) == 81 or cv2.waitKey(10) == 113:
        break

webcam.release()

print("Code Completed!")
