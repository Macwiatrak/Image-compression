# importing packages
%matplotlib inline
import scipy
import numpy as np

# defining max nr of iterations
num_it = 3

# importing an image
A = scipy.misc.face(gray=True)

# assigning nr of rows and columns of A to other matrices
m = A.shape[0]
n = A.shape[1]


# plotting the initial image
from matplotlib import pyplot as plt
#plt.imshow(A,cmap=plt.gray())

# defining k - number of the basis vector which as we will see influences the image compression
k=20

#forming matrices 
V_tilda = np.random.rand(n,k)

#orthogonalizing V
V_tilda, R = scipy.linalg.qr(V_tilda, mode="economic")

# for loop which helps us in image approximation
for i in range(num_it):
    U_temp = np.dot(A,V_tilda)
    
    U_tilda, R = scipy.linalg.qr(U_temp, mode="economic")
    
    V_temp = np.dot(A.transpose(),U_tilda)

    V_tilda, R = scipy.linalg.qr(V_temp, mode="economic")

# Calculating Projected A
A_UV = np.dot(np.dot(U_tilda,(np.dot(np.dot(U_tilda.T,A),V_tilda))),V_tilda.T)

# plotting the images 
from matplotlib import pyplot as plt

# plotting the original image
plt.figure()
plt.imshow(A, cmap=plt.gray())
plt.title('Original image')

# plotting the compressed image approximated through random projection          
plt.figure()
plt.imshow(A_UV, cmap=plt.gray())
plt.title('Compressed image approximated using matrix projection')
