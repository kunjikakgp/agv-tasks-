import cv2
import numpy as np

video = cv2.VideoCapture(r'c:\Users\USER\Downloads\agv1.mp4')  # loading the video
video.set(cv2.CAP_PROP_POS_FRAMES,18)

istrue, old_frame = video.read()  # reading frame by frame ret = True

# optical flow algorithms mostly work in grayscale so converting to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# using shi-tomasi corner detector
# detecting good features to track
feature_params = dict(maxCorners=100, qualityLevel=0.5, minDistance=6, blockSize=7)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # mask specifies region of interest, it is none
# ** unpacks the dictionary
# p0 is a 3d array in numpy

# creating our drawing board
mask = np.zeros_like(old_frame)  # blank image of same size as frames (all pixel values=0)

# using lucas-kanade algorithm (USES INTENSITY DIFFERENCE BETWEEN FRAMES)
# providing parameters
lk_params = dict(winSize=(25, 25), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# termination criteria

# generates random color for rgb for each line that will follow the corners
color = np.random.randint(0, 255, (100, 3))

# while loop to show the video frame by frame
while (1):
    ret, frame = video.read()
    if not ret:
        print("the video has ended")
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
video.release()
