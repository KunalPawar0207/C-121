#Algorithm for INVISIBILITY CLOAK 
# # 1. Capture and store the # background frame. [ This will be # done for some seconds ] 
# # 2. Detect the red colored cloth 
# # using color detection and 
# # segmentation algorithm. 
# # 3. Segment out the red colored 
# # cloth by generating a mask. [ used # in code ] 
# # 4. Generate the final augmented 
# # output to create a magical effect. [ # video.mp4 ]

import cv2
import numpy as np 
import time

cam = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

# To save the output in output.avi
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_file = cv2.VideoWriter("output.avi", fourcc, 20.0, (640,480))

# Capturing background image
for i in range(60):
    ret,bg = cam.read()
bg = np.flip(bg,axis = 1)  

# Capturing the person with cloth and background
while (cam.isOpened()):
    ret,img = cam.read()
    if not ret: break
    img = np.flip(img,axis = 1)
    # Coverting the img from bgr to hsv
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # Generting mask to detect red colour 
    lower_red = np.array([0,120,50])
    upper_red = np.array([10,255,255])
    # Sepreating cloth part from the image
    mask_1 = cv2.inRange(hsv,lower_red,upper_red)
     # Generting mask to detect red colour 
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    # Sepreating cloth part from the image
    mask_2 = cv2.inRange(hsv,lower_red,upper_red)
    # only the cloth part is removed from the image
    mask_1 = mask_1 + mask_2
    # removing noise in the image
    mask_1 = cv2.morphologyEx(mask_1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask_1 = cv2.morphologyEx(mask_1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    # creating mask 2 which contains every thing except the cloth part
    mask_2 = cv2.bitwise_not(mask_1)

    # Keeping the part of the original image without red colour 
    res_1 = cv2.bitwise_and(img,img,mask = mask_2)
    # Replacing the red part of the cloth with background
    res_2 = cv2.bitwise_and(bg,bg,mask = mask_1)

    # Generating the final output 
    final_output  = cv2.addWeighted(res_1,1,res_2,1,0)
    output_file.write(final_output)
    cv2.imshow("magic",final_output)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()