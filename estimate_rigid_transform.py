import numpy as np
from typing import Tuple

# Use for testing ransac
from gen_point_cloud import rodrigues

def main():
   """
   Main entry point for rigid estimation code
   Func. of interest
   =================
   *estimate_rigid_motion: Estimate the pose using RANSAC
   *_estimate_rigid_motion: Estimate the pose without RANSAC
   *test_ransac: test estimation using RANSAC
   """ 
  
   # test_ransac()
 
   P,Q = get_data()
   R,T = estimate_rigid_motion(P,Q,thresh=1,valid=0.2)
   
   numPts,_ = P.shape
   Q_hat = np.dot(R,P.transpose())+np.dot(T[:,np.newaxis],
      np.ones((1,numPts)))
   Q_hat = Q_hat.transpose()
   
   res = np.trace(np.dot((Q-Q_hat).transpose(),(Q-Q_hat)))
  
   # Print pose and frobenius norm of Q-Q_hat
   print('R={0}'.format(R))
   print('T={0}'.format(T)) 
   print('||Q-(R*P+T)||_F^2 = {0}'.format(res))

def estimate_rigid_motion(pt1: np.ndarray, pt2: np.ndarray, 
   numIter: int = 100, minPts: int = 3 , thresh: float = 0.5, 
   valid: float = 0.1) -> Tuple[np.ndarray,np.ndarray]:
   """
   Estimate the rigid motion (R,T) such that X_2 = RX_1 + T using RANSAC
   :input pt1: point cloud, X_1, has dimensions Nx3
   :input pt2: point cloud, X_2, has dimensions Nx3
   :input numIter: (optional) no. of RANSAC iterations
   :input minPts: (optional) no. of points used for initial pose hypothesis
   :input thresh: (optional) threshold used to determine inlier
   :input valid: (optional) min. fraction of points that are inliers 
   :return: the pose
   """
   # Make everything is formatted correctly and
   # have same dimensinos
   h1,w1 = pt1.shape
   h2,w2 = pt2.shape
   assert w1 == 3 and w2 == 3 and h1 == h2 and \
       w1 == w2
 
   bestErr = np.inf
   bestR = np.eye(3)
   bestT = np.zeros(3)

   for ii in range(numIter):   
      
      indx = np.random.choice(h1,minPts,replace=False)
      tmpR,tmpT = _estimate_rigid_motion(pt1[indx,:],pt2[indx,:])
      res = pt2.transpose() - ( np.dot(tmpR,pt1.transpose()) + 
         np.dot(tmpT[:,np.newaxis],np.ones((1,h1))) )
      res = np.sum(res**2,axis=0)
      
      inliers = res < thresh
      numInliers = np.sum(inliers)
      if numInliers > valid*h1:
         tmpR,tmpT = _estimate_rigid_motion(pt1[inliers,:],pt2[inliers,:])
         res = pt2[inliers,:].transpose() - \
            ( np.dot(tmpR,pt1[inliers,:].transpose()) + 
            np.dot(tmpT[:,np.newaxis],np.ones((1,numInliers))) )
         res = np.sum(res**2)/numInliers
         # Retain points with smallest MSE
         if res < bestErr:
            bestErr = res
            bestR = tmpR
            bestT = tmpT

   return bestR,bestT
              
            

def _estimate_rigid_motion(pt1: np.ndarray, pt2: np.ndarray) \
   -> Tuple[np.ndarray,np.ndarray]:
   """
   Estimate the rigid motion (R,T) such that X_2 = RX_1 + T
   :input pt1: point cloud, X_1, has dimensions Nx3
   :input pt2: point cloud, X_2, has dimensions Nx3
   :return: the pose
   """
   h1,w1 = pt1.shape
   h2,w2 = pt2.shape

   # Make everything is formatted correctly and
   # have same dimensinos
   assert w1 == 3 and w2 == 3 and h1 == h2 and \
       w1 == w2
   
   # Center the points
   pt1_centroid = np.mean(pt1,axis=0)
   pt2_centroid = np.mean(pt2,axis=0)
   pt1_cent = pt1 - np.dot( np.ones((h1,1)),
                            pt1_centroid[np.newaxis,:] )
   pt2_cent = pt2 - np.dot( np.ones((h1,1)),
                            pt2_centroid[np.newaxis,:] )
   
   # Perform SVD to get R, using quaternions may avoid handedness
   # issue 
   U,S,V = np.linalg.svd(np.dot(pt1_cent.transpose(),pt2_cent))
   V = V.transpose()
   U = U[:3,:]
   V = V[:3,:]
   # Ensure handedness
   tmp = np.dot(V,U.transpose())
   d = np.linalg.det(tmp)
   U[2,:] *= d
   R = np.dot(V,U.transpose())

   # Estimate T   
   T = np.mean(pt2.transpose()-np.dot(R,pt1.transpose()),axis=1)
   
   return R,T

def test_ransac():
   np.random.seed(0)
   
   # Generate point cloud 
   pt1 = np.random.random((3,100))
   R = rodrigues(np.random.random(3))
   T = np.random.random(3)
   pt2 = np.dot(R,pt1)+np.dot(T[:,np.newaxis],np.ones((1,100)))
   
   # Add Gaussian noise  with high var to a subset of the points to
   # test RANSAC
   outliers = np.random.choice(100,60,replace=False)   
   pt2[:,outliers]= pt2[:,outliers] + \
      np.random.normal(size=(3,60),scale=1) 
   
   print('Ground Truth')
   print('R={0}'.format(R))
   print('T={0}'.format(T))

   Rbad,Tbad = _estimate_rigid_motion(pt1.transpose(),pt2.transpose())   
   print('Estimated w/out RANSAC')
   print('R={0}'.format(Rbad))
   print('T={0}'.format(Tbad))

  
   Rest,Test = estimate_rigid_motion(pt1.transpose(),pt2.transpose(),
      numIter=100, minPts=3, thresh=0.5, valid=0.2)
   print('Estimated w/RANSAC')
   print('R={0}'.format(Rest))
   print('T={0}'.format(Test))

def get_data():
   """
   data from starter code
   """
   P = np.array([[ 0.85715536,  0.19169091,  0.37547468],
            [ 0.36011534,  0.51080372,  0.15273612],
            [ 0.76553819,  0.96956225,  0.07534131],
            [ 0.99562617,  0.51500125,  0.66700672],
            [ 0.75873823,  0.57802293,  0.68742226],
            [ 0.40803516,  0.86300463,  0.84899731],
            [ 0.3681771 ,  0.11123281,  0.56888489],
            [ 0.42113594,  0.01455699,  0.02298089],
            [ 0.41690264,  0.49362725,  0.12992772],
            [ 0.7648861 ,  0.99328892,  0.131986  ],
            [ 0.24313837,  0.49029503,  0.45695072],
            [ 0.52637675,  0.97293938,  0.55972334],
            [ 0.84064045,  0.1832065 ,  0.82720688],
            [ 0.82169952,  0.66711466,  0.32593019],
            [ 0.24323087,  0.1992479 ,  0.1700658 ],
            [ 0.6131605 ,  0.51652155,  0.30193314],
            [ 0.1025926 ,  0.87449354,  0.2651951 ],
            [ 0.8714256 ,  0.46842985,  0.84525766],
            [ 0.83809381,  0.28522291,  0.06991397],
            [ 0.76453123,  0.97233898,  0.93372074]])

   Q = np.array([[ 0.40901504,  0.83198029,  0.87697249],
            [ 0.24356397,  0.24470076,  0.71503754],
            [-0.18577546,  0.30922866,  1.15358153],
            [ 0.43260626,  0.75349715,  1.3263927 ],
            [ 0.49888288,  0.52382659,  1.26839902],
            [ 0.60616086,  0.07094228,  1.38545005],
            [ 0.7710987 ,  0.47837833,  0.71313771],
            [ 0.3663448 ,  0.58408716,  0.34549392],
            [ 0.21444126,  0.30117119,  0.71677242],
            [-0.15225962,  0.29424925,  1.20281878],
            [ 0.53620743,  0.15779029,  0.83104238],
            [ 0.28042234,  0.10685548,  1.33672152],
            [ 0.77861652,  0.81812   ,  1.13699047],
            [ 0.14504543,  0.52761744,  1.13663154],
            [ 0.45264174,  0.33027575,  0.4701034 ],
            [ 0.27256995,  0.44559883,  0.9274591 ],
            [ 0.24052756, -0.17745923,  0.89671581],
            [ 0.6404062 ,  0.67733176,  1.34604188],
            [ 0.12568749,  0.76576281,  0.74348215],
            [ 0.4966933 ,  0.29664203,  1.67406394]])
   
   return P,Q

if __name__ == '__main__':
   main()
