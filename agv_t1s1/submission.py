"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2 as cv
from numpy.linalg import svd,norm
import matplotlib.pyplot as plt


"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # replace pass by your implementation
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
    return F


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    # replace pass by your implementation
    corrpts = []
    h, w = im2.shape[:2]
    ws =5
    
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


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(K1, F, K2):
    # replace pass by your implementation
    E=K2.T@F@K1
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(ptsim1, ptsim2, P1, P2):
    # replace pass by your implementation
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


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    # Compute optical centers CORRECTLY (remove t2 normalization)
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
   
    
    # Compute baseline direction CORRECTLY (c2 - c1)
    r1 = (c1- c2).flatten()
    r1 = r1/norm(r1)
    print("r1 is ",r1)    
    # Use original camera's Y-axis for alignment
    z= R1[2, :] 
    r2 = np.cross(z.T, r1)
    r2 /= norm(r2)
    
    # Complete orthogonal basis
    r3 = np.cross(r2, r1)
    R = np.vstack([r1, r2, r3]).T
    print(R)
    KR=K2@R
    KR1=K1@R1
    invKR1=np.linalg.inv(KR1)
    # Compute rectification matrices
    M1 = KR@invKR1
    KR2=K2@R2
    invKR2=np.linalg.inv(KR2)
    M2= KR@invKR2
    
    return M1, M2


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
