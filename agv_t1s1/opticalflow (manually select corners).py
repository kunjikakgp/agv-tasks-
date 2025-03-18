import cv2 as cv
import numpy as np

selected_points = []
click_active = True

# Mouse callback function
def select_points(event, x, y, flags, param):
    global selected_points, click_active
    if event == cv.EVENT_LBUTTONDOWN and click_active:
        cv.circle(old_frame, (x, y), 5, (0, 255, 0), -1)
        selected_points.append([x, y])
        cv.imshow('Select Points', old_frame)

video = cv.VideoCapture(r'c:\Users\USER\Downloads\agv1.mp4') 
video.set(cv.CAP_PROP_POS_FRAMES, 18)

istrue, old_frame = video.read() 
if not istrue:
    print("Error loading video")
    exit()

# SELECTING CORNERS IN FIRST FRAME 
cv.namedWindow('Select Points')
cv.setMouseCallback('Select Points', select_points)
cv.imshow('Select Points', old_frame)

print("Click points to track (press ENTER when done, ESC to cancel)")
while True:
    key = cv.waitKey(1) & 0xff
    if key == 13:  # ENTER key
        break
    elif key == 27:  # ESC key
        cv.destroyAllWindows()
        exit()

cv.destroyWindow('Select Points')
click_active = False

# Converting  selected points to numpy array
if len(selected_points) == 0:
    print("No points selected!")
    exit()
    
p0 = np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)
oldgray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(old_frame)
color = np.random.randint(0, 255, (100, 3))

lkparams = dict(winSize=(25, 25), maxLevel=2,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    ret, frame = video.read()
    if not ret:
        print("The video has ended")
        break
        
    framegray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(oldgray, framegray, p0, None, **lkparams)

    # Select good points
    if p1 is not None and len(p1) > 0:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.arrowedLine(mask, (int(c), int(d)), (int(a), int(b)), 
                            color[i].tolist(), tipLength=0.2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        
    img = cv.add(frame, mask)
    cv.imshow('Optical Flow', img)
    
    key = cv.waitKey(30) & 0xff
    if key == 27:
        break

    # Update previous frame and points
    oldgray = framegray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
video.release()
