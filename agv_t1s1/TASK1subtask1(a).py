import cv2 as cv
import numpy as np
# using lucas-kanade algorithm (USES INTENSITY DIFFERENCE BETWEEN FRAMES)
video = cv.VideoCapture(r'c:\Users\USER\Downloads\agv1.mp4') 
video.set(cv.CAP_PROP_POS_FRAMES,18)

istrue, old_frame = video.read() 
if not istrue:
    print("error loading video")
    exit 

# optical flow algorithms mostly work in grayscale so converting to grayscale
oldgray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# using shi-tomasi corner detector
# detecting good features to track
feature_params = dict(maxCorners=100, qualityLevel=0.5, minDistance=6, blockSize=7)

p0 = cv.goodFeaturesToTrack(oldgray, mask=None, **feature_params) 
# ** unpacks the dictionary
# p0 is a 3d array in numpy

# creating our drawing board
mask = np.zeros_like(old_frame) 

# providing lukas kanade parameters
lkparams = dict(winSize=(25, 25), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# generates random color for rgb for each line that will follow the corners
color = np.random.randint(0, 255, (100, 3))
while (1):
    ret, frame = video.read()
    if not ret:
        print("the video has ended")
        break
    framegray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(oldgray, framegray, p0, None, **lkparams)

    # Select good points
    if p1 is not None and len(p1)>0:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.arrowedLine(mask, (int(c), int(d)), (int(a), int(b)), color[i].tolist(),tipLength=0.2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    oldgray = framegray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
video.release()
