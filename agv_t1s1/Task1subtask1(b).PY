import cv2
import numpy as np

video = cv2.VideoCapture(r'c:\Users\USER\Downloads\agv1.mp4') 
video.set(cv2.CAP_PROP_POS_FRAMES, 18)  # Starting  from frame 18

ret, oldframe = video.read()
if not ret:
    print("Error reading video")
    exit()

oldgray = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY) 

# HSV mask for visualization
hsv = np.zeros_like(oldframe)
hsv[..., 1] = 255  # Set saturation to max

while True:
    ret, frame = video.read()
    if not ret:
        print("The video has ended")
        break

    framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

    # using Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(oldgray, framegray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to polar coordinates (angle and magnitude)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Encoding  angle as hue (direction of motion)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Normalize angle to [0, 180]
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Normalize magnitude to [0, 255]

    # Convert HSV to BGR 
    flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Dense Optical Flow (Farneback)', flow)

    # Press 'Esc' to exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

    # Update previous frame
    oldgray = framegray.copy()

video.release()
cv2.destroyAllWindows()
