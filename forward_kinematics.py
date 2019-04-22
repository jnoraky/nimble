import numpy as np 
from typing import List

class RobotUR5(object):
   """
   Robot class used for forward kinematics problem
   """
   # DH-parameters for UR5
   # for simplicity everything all distances to 1
   alpha_vec = [ 0, np.pi/2, 0, 0, np.pi/2, -np.pi/2 ]
   a_vec = [ 0, 0, 1, 1, 0, 0 ]
   d_vec = [ 1, 0, 0, 1, 1, 1 ]

   def __init__(self, tool_transform: np.ndarray) -> None:
      """
      Initialize the robot class 
      :input tool_transform: 4x4 matrix specifying the tool transform
      (probably used for inverse problem)
      """
      self.tool_transform = tool_transform
   
   @staticmethod
   def get_transform(alpha,a,d,theta):
      """
      Convert the DH parameters to a transformation
      """
      T = np.zeros((4,4))
      T[0,:] = [ np.cos(theta), -np.sin(theta), 0, a ]
      T[1,:] = [ np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), \
         -np.sin(alpha), -np.sin(alpha)*d ]      
      T[2,:] = [ np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), \
         np.cos(alpha), np.cos(alpha)*d ]
      T[3,:] = [ 0, 0, 0, 1 ]
      return T

   def forward_kinematics(self, joint_angles: List) -> np.ndarray:
      """
      Get transform of end-effector relative to robot base
      :input joint_angles: list of joint angles
      :return: 4x4 transform
      """
      T = np.eye(4) 
      for alpha,a,d,theta in zip(RobotUR5.alpha_vec,RobotUR5.a_vec,
                                 RobotUR5.d_vec,joint_angles):
         T = np.dot( T, RobotUR5.get_transform(alpha,a,d,theta) )
      return T

if __name__ == '__main__':
   # Testing with arbitrary angles 
   rob = RobotUR5(np.eye(4,4))
   transform = rob.forward_kinematics([ (np.pi/4)/i for i in range(1,7) ])
   print("The final transform T is: ")
   print(transform)
   print("T[:3,:3]^T*T[:3,:3] should be close to identity :")
   print(np.dot(transform[:3,:3].transpose(),transform[:3,:3]))     
   
