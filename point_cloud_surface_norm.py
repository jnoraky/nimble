import numpy as np
from typing import Tuple

from gen_point_cloud import rodrigues

def main():
   """
   Main entry point to estimate surface normal 
   Functions of interest:
   * estimate_surface_normal - estimate the surface normal given a point 
   cloud and point using RANSAC
   * _surface_normal - estimate the surface normal given point cloud patch,
   assumes all points are inlier and used by estimate_surface_normal func
   * test_surface_normal - test code using synthetic data, showing how to 
   run the functions
   """

   test_surface_normal()


def estimate_surface_normal(pt_cloud: np.ndarray, pt: np.ndarray, 
   patch_radius: int = 10, numIter: int = 100, thresh: float = 1,
   valid: float = 0.2) -> np.ndarray:
   """
   Estimate the surface of normal of a point cloud at a given point
   using RANSAC
   :input pt_cloud: point cloud with dim Nx6 (x,y,z,r,g,b)
   :input pt: point to estimate surface normal at
   :input patch_radius: radius of region used to estimate the surface normal
   :input numIter: no. RANSAC iterations
   :input thresh: threshold used to determine RANSAC inlier 
   :input valid: min. fraction of points that must be inliers
   :return: the surface normal
   """ 
   h_pc,w_pc = pt_cloud.shape

   assert w_pc == 6, 'pt_cloud must be Nx6'
   assert len(np.squeeze(pt)) == 3, 'pt must be a 3 vector'

   dist_to_pt = pt_cloud[:,:3] - np.dot(np.ones((h_pc,1)), pt[np.newaxis,:])
   dist_to_pt = np.sqrt(np.sum(dist_to_pt**2,axis=1))
   assert len(dist_to_pt) == h_pc, 'Something went wrong with dims' 

   patch = pt_cloud[dist_to_pt<patch_radius,:3]
   numPts,_ = patch.shape
   assert numPts >= 3, 'Need at least 3 points in patch to proceed'
   A = np.ones((numPts,4))
   A[:,:3] = patch
   best_normal = np.zeros(3)
   min_mse = np.inf
   for ii in range(numIter):
      indx = np.random.choice(numPts,3,replace=False)
      temp_normal,tmp_coeffs = _surface_normal(patch[indx,:])
      err = np.dot(A,tmp_coeffs)**2
      
      inliers = err < thresh
      numInliers = np.sum(inliers)
      if numInliers > valid*numPts:
         tmp_normal,tmp_coeffs = _surface_normal(patch[inliers,:])
         err = np.sum(np.dot(A,tmp_coeffs)**2)/numInliers
         if err < min_mse:
            min_mse = err
            best_normal = tmp_normal

   return best_normal
         

def _surface_normal(pts: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
   """
   Calculate surface normal without RANSAC
   :input pts: point cloud region with dim Nx3 
   :return: surface normal and plane coefficients
   """
   h,w = pts.shape
   assert w == 3
   
   A = np.ones((h,4))
   A[:,:3] = pts
   # eigh returns eigenvectors sorted by eigenvaleus in ascending order
   eigenvals, eigenvecs = np.linalg.eigh(np.dot(A.transpose(),A))
   
   coeffs = eigenvecs[:,0]
   normal = coeffs[:3]/norm2(coeffs[:3])
   if normal[2] > 0:
      normal *= -1
 
   return normal,coeffs

def norm2(vec: np.ndarray) -> float:
   """
   return norm of vector
   :input vec: input vector
   :return 2-norm
   """
   return np.sqrt(np.sum(vec**2))

def test_surface_normal():
   """
   Test surface normal estimation code using synthetic data
   """
   
   # Make data
   np.random.seed(0)
   X,Y = np.meshgrid(range(10),range(10))
   Z = 1*np.ones(X.shape)

   pts = np.zeros((3,100))
   pts[0,:] = X.flatten()
   pts[1,:] = Y.flatten()
   pts[2,:] = Z.flatten()

   R = rodrigues(np.random.random(3))
   pts2 = np.dot(R,pts)
   normal = -1*R[:,2]

   pt_cloud = np.zeros((100,6))
   pt_cloud[:,:3] = pts2.transpose()
   pt = pts2[:,0]
  
   ############### Test noise free case ##########################
   print('Testing noise-free case')
   normal_hat = estimate_surface_normal(pt_cloud,pt,thresh=np.inf)
   print('ground truth = {0}'.format(normal))
   print('estimated = {0}'.format(normal_hat))

   ############### Test the noise case #########################
   print('\nTesting noise case')
   # Corrupt some subset
   outliers = np.random.choice(100,50,replace=False) 
   pts2[:,outliers] += np.random.normal(size=(3,50),scale=1)
   pt_cloud[:,:3] = pts2.transpose()

   normal_bad = estimate_surface_normal(pt_cloud,pt,thresh=np.inf)
   normal_hat = estimate_surface_normal(pt_cloud,pt,thresh=1e-5)

   print('ground truth = {0}'.format(normal))
   print('estimated w/out ransac = {0}'.format(normal_bad))
   print('estimated w/ransac= {0}'.format(normal_hat))

   ############### Test distance threshold #####################
   print('\nTesting the distance threshold')
   R = rodrigues(np.random.random(3))
   pts3 = np.dot(R,pts)
   pts3[2,:] += 50
   
   pt_cloud = np.zeros((200,6))
   pt_cloud[:100,:3] = pts2.transpose()
   pt_cloud[100:,:3] = pts3.transpose()

   normal_bad = estimate_surface_normal(pt_cloud,pt,patch_radius=np.inf,\
      thresh=1e-5)
   normal_hat = estimate_surface_normal(pt_cloud,pt,patch_radius=20,
      thresh=1e-5)

   print('ground truth = {0}'.format(normal))
   print('estimated w/out distance thresh = {0}'.format(normal_bad))
   print('estimated w/distance thresh= {0}'.format(normal_hat))



if __name__ == '__main__':
   main()
