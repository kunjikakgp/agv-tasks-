import numpy as np
import cv2 as cv
from numpy.linalg import svd,norm,inv
from scipy.signal import convolve2d

def rectify(K1, K2, R1, R2, t1, t2):
    KR1=K1@R1
    inv1=np.linalg.inv(KR1)
    KT1=K1@t1
    c1 = inv1@KT1
    print("c1 is \n",c1)
    KR2=K2@R2
    inv2=np.linalg.inv(KR2)
    KT2=K2@t2
    c2 = inv2@KT2
    print("c2 is \n",c2)
   

    r1 = (c1- c2).flatten()
    r1 = r1/norm(r1)
    print("r1 is ",r1)    
  
    z= R1[2, :] 
    r2 = np.cross(z.T, r1)
    r2 /= norm(r2)
    

    r3 = np.cross(r2, r1)
    R = np.vstack([r1, r2, r3]).T
    print(R)
    KR=K2@R
    KR1=K1@R1
    invKR1=np.linalg.inv(KR1)
   
    M1 = KR@invKR1
    KR2=K2@R2
    invKR2=np.linalg.inv(KR2)
    M2= KR@invKR2
    
    return M1,M2,c1,c2
def disparity(im1, im2, max_disp, win_size):
    height, width = im1.shape
    
    dispM = np.zeros((height, width))
    
    #using SSD
    window = np.ones((win_size, win_size))
    ssd_map = np.zeros((height, width, max_disp))
    
    # Iterate over all possible disparities
    for d in range(max_disp):
        shifted_im2 = np.roll(im2, d, axis=1)
        
        # managing out of bounds errors
        shifted_im2[:, :d] = 0
        squared_diff = (im1 - shifted_im2) ** 2
        
        # Applying convolution
        ssd_map[:, :, d] = convolve2d(squared_diff, window, mode='same', boundary='symm')
    
    # For each pixel, find the disparity with the minimum SSD
    for y in range(height):
        for x in range(width):
            dispM[y, x] = np.argmin(ssd_map[y, x, :]) 
    return dispM
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    result=rectify(K1, K2, R1, R2, t1, t2)
    c1=result[2]
    c2=result[3]
    b = np.linalg.norm(c1 - c2)
    f = K1[0, 0]
    depthM = np.zeros_like(dispM, dtype=np.float32)
    nonzero_mask = dispM > 0
    depthM[nonzero_mask] = (b * f) / dispM[nonzero_mask]
    return depthM
im1 = cv.imread(r"c:\Users\USER\Downloads\im1.png", cv.IMREAD_GRAYSCALE) 
im2 = cv.imread(r"c:\Users\USER\Downloads\im2.png", cv.IMREAD_GRAYSCALE)
intrinsic=np.load(r'c:\Users\USER\Downloads\intrinsics.npz')
K1=intrinsic[intrinsic.files[0]]
K2=intrinsic[intrinsic.files[1]]
data=np.load('dataextrinsics.npz')
print(data.files)
R1=data['R1']
R2=data['R2']
t1=data['t1']
t2= data['t2']
dispM=disparity(im1, im2, 16, 5)
depthM=get_depth(dispM, K1, K2, R1, R2, t1, t2)
#color coding depths 
# Normalize the depth map to the range 0-255
depthM_normalized = cv.normalize(depthM, None, 0, 255, cv.NORM_MINMAX)
depthM_display = depthM_normalized.astype(np.uint8)

depthM_colormap = cv.applyColorMap(depthM_display, cv.COLORMAP_JET)

cv.imshow("Depth Map", depthM_colormap)
cv.waitKey(0)
cv.destroyAllWindows()
