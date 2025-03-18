#importing relevant libraries
import numpy as np
import cv2 as cv
from numpy.linalg import svd,norm
import matplotlib.pyplot as plt
#computing the fundamental matrix
#pts1 and pts2 are the given points in some correspondance 
def eight_point(pts1,pts2,M):
     # Normalize points
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    pts1_norm = np.hstack((pts1, np.ones((pts1.shape[0], 1)))) @ T.T
    pts2_norm = np.hstack((pts2, np.ones((pts2.shape[0], 1)))) @ T.T

    # Construct A matrix
    A = np.zeros((pts1_norm.shape[0], 9))
    for i in range(pts1_norm.shape[0]):
        x1, y1, _ = pts1_norm[i]
        x2, y2, _ = pts2_norm[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    # Solve for F using SVD
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    # Enforce rank 2 constraint
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V

    # Refine F (helper function provided)
    #F = refineF(F, pts1_norm[:, :2], pts2_norm[:, :2])

    # Unnormalize F
    F = T.T @ F @ T
    print(F)
    return F
#finding corresponding points 
def epipolar_correspondences(im1, im2, F, pts1, window_size=5):
    corrpts = []
    h, w = im2.shape[:2]
    ws = window_size
    
    for pt in pts1:
        # Convert to homogeneous coordinates
        x1, y1 = int(pt[0]), int(pt[1])
        pt_hom = np.array([x1, y1, 1])
        
        # Compute epipolar line (ax + by + c = 0)
        l = F @ pt_hom
        a, b, c = l
        
        # Generate candidate points along the epipolar line
        candidates = []
        for x in range(w):
            y = int((-a*x - c)/b) if b != 0 else 0
            if 0 <= y < h:
                candidates.append((x, y))
        
        # Handle vertical lines
        if not candidates and a != 0:
            for y in range(h):
                x = int((-b*y - c)/a) if a != 0 else 0
                if 0 <= x < w:
                    candidates.append((x, y))

        best_match = None
        best_score = float('inf')
        
        # Extract template window from im1 (with boundary checks)
        y1_min = max(y1 - ws, 0)
        y1_max = min(y1 + ws + 1, h)
        x1_min = max(x1 - ws, 0)
        x1_max = min(x1 + ws + 1, w)
        template = im1[y1_min:y1_max, x1_min:x1_max]
        
        if template.size == 0:
            corrpts.append((-1, -1))  # Invalid marker
            continue
            
        for x2, y2 in candidates:
            # Extract window from im2
            y2_min = max(y2 - ws, 0)
            y2_max = min(y2 + ws + 1, h)
            x2_min = max(x2 - ws, 0)
            x2_max = min(x2 + ws + 1, w)
            
            window = im2[y2_min:y2_max, x2_min:x2_max]
            
            # Skip mismatched window sizes
            if window.shape != template.shape:
                continue
                
            # Compute similarity
            score = np.mean((template.astype(float) - window.astype(float)) ** 2)
            
            if score < best_score:
                best_score = score
                best_match = (x2, y2)
        
        corrpts.append(best_match if best_match else (-1, -1))
    
    return np.array(corrpts)
#compute essential matrix
def essential_matrix(K1,F,K2):
    E=K2.T@F@K1
    return E
#finding candidates for p2(helper function)
def camera2(E):
    """Decompose essential matrix into 4 possible camera configurations"""
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # Ensure rotation matrices are proper (det=1)
    if np.linalg.det(U @ W @ Vt) < 0:
        W = -W
    
    # Get translation vector (unit norm)
    t = U[:, 2]
    t /= np.linalg.norm(t)  # Unit vector
    
    # Create 4 configurations
    M2s = np.zeros((3, 4, 4))
    for i in range(4):
        R = U @ W @ Vt if i < 2 else U @ W.T @ Vt
        sign = 1 if i % 2 == 0 else -1
        
        # Ensure det(R) = 1
        if np.linalg.det(R) < 0:
            R *= -1
            
        M2s[:, :, i] = np.hstack([R, sign * t.reshape(-1, 1)])
    
    return M2s
#finding real world 3d points  using triangulation
def triangulate(ptsim1, ptsim2, P1, P2):
    N = ptsim1.shape[0]
    pts3d = np.zeros((N, 3))
    
    for i in range(N):
        x1, y1 = ptsim1[i][:2]
        x2, y2 = ptsim2[i][:2]
        
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])
        
        _, _, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]
        X = X_homogeneous[:3] / X_homogeneous[3]
        pts3d[i] = X
    
    return pts3d
#finding P2
def extrinsic(P1,P2_candidates,ptsim1,ptsim2):
    best=None
    maxi=0
    for P2 in P2_candidates:
        pts3d = triangulate(ptsim1,ptsim2,P1,P2)
        valid_points = np.sum(pts3d[:, 2] > 0)#(you can also concvert it to loop)
        if valid_points > maxi:
            maxi = valid_points
            best = P2
    return best
def reprojection_error(points3D, points2D, projection_matrix):
    N = points3D.shape[0]
    
    # Convert 3D points to homogeneous coordinates
    points3D_homogeneous = np.hstack((points3D, np.ones((N, 1))))
    
    # Project 3D points to 2D
    projected_points = projection_matrix @ points3D_homogeneous.T
    projected_points = projected_points.T
    projected_points = projected_points[:, :2] / projected_points[:, 2:]
    
    # Calculate Euclidean distance between observed and projected points
    errors = np.linalg.norm(points2D - projected_points, axis=1)
    
    # Calculate RMS error
    rms_error = np.sqrt(np.mean(errors**2))
    
    return rms_error
#loading the data and visualising computations
#load the images
im1=cv.imread(r"c:\Users\USER\Downloads\im1.png")
im2=cv.imread(r"c:\Users\USER\Downloads\im2.png")
height,width=im1.shape[:2]
M=max(height,width)
#load the given points 
givenpoints=np.load(r"c:\Users\USER\Downloads\some_corresp.npz")
pts1=givenpoints[givenpoints.files[0]]
pts2=givenpoints[givenpoints.files[1]] #Nx2 matrix
#find the fundamental matrix
F=eight_point(pts1,pts2,M)
#find correspondences
data=np.load(r"c:\Users\USER\Downloads\temple_coords.npz")
impts1=data[data.files[0]]
impts2=epipolar_correspondences(im1, im2, F, impts1, window_size=5)
#compute essential matrix
intrinsic=np.load(r'c:\Users\USER\Downloads\intrinsics.npz')
K1=intrinsic[intrinsic.files[0]]
K2=intrinsic[intrinsic.files[1]]
E=essential_matrix(K1,F,K2)
#computing projection matrices
P1=np.concatenate([K1, np.zeros((3, 1))], axis=1)
M2s = camera2(E)
P2_candidates = []
for i in range(4):
    extrinsic_matrix = M2s[:, :, i]
    projection_matrix = K2 @ extrinsic_matrix  
    P2_candidates.append(projection_matrix)
P2_candidates = np.array(P2_candidates)
P2 = extrinsic(P1, P2_candidates, impts1, impts2)
#reconstructing
pts3d=triangulate(impts1, impts2, P1, P2)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming pts3d is your Nx3 array of 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set title
ax.set_title('3D Point Cloud Reconstruction')

# Show the plot
plt.show()
# Assuming R1, t1 are identity rotation and zero translation for first camera
R1 = np.eye(3)
t1 = np.zeros((3, 1))



# Extract R2 and t2 from the correct P2
P2_new = np.linalg.inv(K2) @ P2 
R2 = P2_new[:, :3]
if np.linalg.det(R2) < 0:
     R2 *= -1
t2_homogeneous = P2[:, 3]
t2 = (np.linalg.inv(K2) @ t2_homogeneous)
t2 = t2.reshape((3, 1))



# Save extrinsic parameters
np.savez('dataextrinsics.npz', R1=R1, R2=R2, t1=t1, t2=t2)




