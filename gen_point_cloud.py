import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_point_cloud():
   """
   Show how gen_point_cloud is called
   Functions of interest:
   * gen_point_cloud - Make a point cloud given rgb image, depth map, 
   intrinsics, and extrinsics parameter
   * taitbryan - specify rotation using taitbryan intrinsic angles
   * rodrigues - specify rotation using rodrigues formula
   """ 
   dmap = np.load('0.npy')
   rgb = cv2.imread('rgb0.png',-1)
   
   # Specify intrinsics
   K = np.zeros((3,3))
   K[0,:] = [6.196433105468750000e+02,0.000000000000000000e+00,\
      3.083635253906250000e+02]
   K[1,:] = [0.000000000000000000e+00,6.192536010742187500e+02,\
      2.417334289550781250e+02]
   K[2,:] = [0.000000000000000000e+00,0.000000000000000000e+00,\
      1.000000000000000000e+00]

   # Specify the extrinsics
   # Assumed to be R, T, where R is parameterized by Tait-Bryan
   extrinsics = np.array([0.04672947, -0.77292125, -0.58312806,  \
      0.03052898,  2.93777692, 1.10868356])

   pt_cloud = gen_point_cloud(rgb, dmap, K, extrinsics)

   # Test that point cloud maps back into an image that is the same as
   # the input image
   R = taitbryan(extrinsics[:3])
   T = extrinsics[3:]
   _xyz = pt_cloud[:,:3].transpose()
   _,numPts = _xyz.shape
   _rgb = pt_cloud[:,3:].transpose()
   pts2 = np.dot(R,_xyz)+np.dot(T[:,np.newaxis],np.ones((1,numPts)))
   pts2 = np.dot(K,pts2)
   img_hat = np.zeros(rgb.shape)
   for ii in range(numPts):
      x,y,z = pts2[:,ii]
      try:
         x = int(round(x/z))
         y = int(round(y/z))
         img_hat[y,x,:] = _rgb[:,ii]
      except:
         pass

   # Check if reprojected point cloud matches RGB image for non-zero 
   # depth values
   err = (rgb[:,:,0].astype(np.float)-img_hat[:,:,0]).flatten()
   red_check = (np.sum(err*dmap.flatten()!=0))
   err = (rgb[:,:,1].astype(np.float)-img_hat[:,:,1]).flatten()
   blue_check = (np.sum(err*dmap.flatten()!=0))
   err = (rgb[:,:,2].astype(np.float)-img_hat[:,:,2]).flatten()
   green_check = (np.sum(err*dmap.flatten()!=0))
   assert (red_check == 0) and (blue_check == 0) and (green_check==0)

   # plt.figure()
   # plt.imshow(dmap)
   # plt.figure()
   # plt.imshow(img_hat.astype(np.uint8))
   # plt.show()   
 
   return pt_cloud

def gen_point_cloud(rgb: np.ndarray, dmap: np.ndarray, intr: np.ndarray, 
   ext: np.ndarray) -> np.ndarray:
   """
   Get the point cloud given a depth map (x,y,z_c) in the world coordinate 
   frame (X_w,Y_w,Z_w) where we assume that:
               z_c*(x,y,1)^T = K[ R | T ]*(X_w,Y_w,Z_w,1)^T
   
   :input rgb: rgb image
   :input dmap: depth map
   :input intr: intrinsic parameters 3x3
   :input ext: extrinsic parameters, 6x1, where ASSUMING first 3 are 
   rotation and last 3 are translation. Assuming Euler angles
   :return: point cloud in Nx6 (X_w,Y_w,Z_w,r,g,b)
   """

   h,w = dmap.shape
   pt_cloud = np.zeros((h*w,6),dtype=np.float)
   pt_cloud[:,3] = rgb[:,:,0].flatten()
   pt_cloud[:,4] = rgb[:,:,1].flatten()
   pt_cloud[:,5] = rgb[:,:,2].flatten()

   # Get the points in the camera's frame, assuming no-zero skew
   xx,yy = np.meshgrid(range(w),range(h))
   fx,skew,cx = intr[0,:]
   _,fy,cy = intr[1,:]
   
   Z = dmap.flatten()
   Y = Z*(yy.flatten()-cy)/fy
   X = (Z*(xx.flatten()-cx)-skew*Y)/fx
   
   pts_cam = np.zeros((3,h*w))
   pts_cam[0,:] = X
   pts_cam[1,:] = Y
   pts_cam[2,:] = Z 
   
   # Get the world coorinates
   # ASSUME extrinsics are given as [R, T]
   # There are many ways to specify rotation -- we provide Tait-Bryan 
   # formation and Rodrigues below
   R = taitbryan(ext[:3]) # or can use Rodrigues (see below)
   T = ext[3:]
   # Go from Camera to Wrold
   pts = np.dot(R.transpose(), pts_cam-
      np.dot(T[:,np.newaxis],np.ones((1,h*w))))
   
   pt_cloud[:,:3] = pts.transpose()

   return pt_cloud

################ Utils Functions ################################
def rodrigues(omega):
   angle = np.sqrt(np.sum(omega**2))
   if angle == 0:
      return np.eye(3)
   a,b,c = omega/angle
   K = np.zeros((3,3))
   K[0,:] = [0, -c, b]
   K[1,:] = [c, 0, -a]
   K[2,:] = [-b, a, 0]
   return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*np.dot(K,K)

def taitbryan(ypr):
   """
   Rotation specified by yaw pitch roll : Rz(yaw)*Ry(pitch)*Rx(roll)
   """
   # Rotation parameters
   yaw = ypr[0]
   pitch = ypr[1]
   roll = ypr[2]
   # Roll
   Rx = np.zeros((3,3))
   Rx[0,:] = [1,0,0]
   Rx[1,:] = [0,np.cos(roll),-np.sin(roll)]
   Rx[2,:] = [0,np.sin(roll),np.cos(roll)]
   # Pitch
   Ry = np.zeros((3,3))
   Ry[0,:] = [np.cos(pitch),0,np.sin(pitch)]
   Ry[1,:] = [0,1,0]
   Ry[2,:] = [-np.sin(pitch),0,np.cos(pitch)]
   # Yaw
   Rz = np.zeros((3,3))
   Rz[0,:] = [np.cos(yaw),-np.sin(yaw),0]
   Rz[1,:] = [np.sin(yaw),np.cos(yaw),0]
   Rz[2,:] = [0,0,1]
   
   return np.dot(Rz,np.dot(Ry,Rx))

  
if __name__ == '__main__':
   get_point_cloud()
