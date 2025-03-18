import cv2
import numpy as np

# Load video
video = cv2.VideoCapture(r'c:\Users\USER\Downloads\agv1.mp4')
video.set(cv2.CAP_PROP_POS_FRAMES, 18)  # Start from frame 18

# Read the first frame
ret, old_frame = video.read()
if not ret:
    print("Error reading video")
    exit()

# Convert first frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Step size for arrows (adjust for density)
step = 15  

while True:
    ret, frame = video.read()
    if not ret:
        print("The video has ended")
        break

    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 10, 3, 15, 1.2, 0)

    # Draw arrows on a grid
    for y in range(0, frame.shape[0], step):
        for x in range(0, frame.shape[1], step):
            dx, dy = flow[y, x]  # Motion vector

            # Compute arrow start and end points
            start_point = (x, y)
            end_point = (int(x + dx), int(y + dy))

            # Draw arrow
            cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 1, tipLength=0.3)

    # Show the frame with arrows
    cv2.imshow("Optical Flow (Arrows)", frame)

    # Press 'Esc' to exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

    # Upda
