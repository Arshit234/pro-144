import cv2
import numpy as np

facecam = cv2.VideoCapture(0)  

background_image = cv2.imread('download.jpeg')

def add_background_to_frame(frame, background_image):
    
    background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

    
    frame_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    frame_mask_inv = cv2.bitwise_not(frame_mask)

    
    background = cv2.bitwise_and(background_image, background_image, mask=frame_mask)

    
    frame_roi = cv2.bitwise_and(frame, frame, mask=frame_mask_inv)

    
    final_frame = cv2.add(background, frame_roi)

    return final_frame

while True:
    ret, frame = facecam.read()
    if not ret:
        break

    
    frame_with_background = add_background_to_frame(frame, background_image)

    
    cv2.imshow('Facecam with Background', frame_with_background)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

facecam.release()
cv2.destroyAllWindows()



