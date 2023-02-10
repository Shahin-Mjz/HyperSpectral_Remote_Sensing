import cv2
import numpy as np



# Noise Removal Block:

# Adaptive Median Filter:
# The minimum kernel size is s.
# The maximum kernel size is sMax.
# In Adaptive Median Filter s(minimum kernel size) should be 3 because if this be 1 we will have a kernel with 1-by-1 size and calculation of the median will be meaningless.
# Increasing the sMax leads to the improvement of output.
# Thus the sMax is the variable hyperparameter in Optimization Algorithm.

# parameters: 3 <= SMax and must be odd.
def adaptive_median_filter(img_gray: np.ndarray, s=3, sMax=11) -> np.ndarray:


    H, W = img_gray.shape
    a = sMax//2
    
    padded_img = np.pad(img_gray, a, mode='constant')
    f_img = np.zeros(padded_img.shape)

    for i in range(a, H + a + 1):
        for j in range(a, W + a + 1):
            
            f_img[i, j] = Lvl_A(padded_img, i, j, s, sMax)

    return np.array(f_img[a: -a, a: -a], dtype=np.uint8)


def Lvl_A(mat, x, y, s, sMax):

    window = mat[x - (s//2): x + (s//2) + 1, y - (s//2): y + (s//2) + 1]
    Zmin = np.min(window)
    
    win_vect = window.reshape(-1)
    Zmed = np.sort(win_vect)[len(win_vect)//2] # O(nlog(n))
    
    Zmax = np.max(window)

    if Zmin < Zmed < Zmax:
        return Lvl_B(window, Zmin, Zmax, Zmed)
    else:
        s += 2 
        if s <= sMax:
            return Lvl_A(mat, x, y, s, sMax)
        else:
             return Zmed

            
def Lvl_B(window, Zmin, Zmax, Zmed):

    h, w = window.shape
    Zxy = window[h//2, w//2]

    if Zmin < Zxy < Zmax:
        return Zxy
    else:
        return Zmed


# Gaussian Blur Filter:

# Parameters:

# kSize: that values of it must be odd and these values can be different values.
# sigmaX and sigmaY: These can be negative or zero or positive values.
def gaussian_blur_filter(img_gray: np.ndarray, kSize: tuple =(1, 1), sigmax: float =0, sigmay: float =0) -> np.ndarray:
    return cv2.GaussianBlur(img_gray, ksize=kSize, sigmaX=sigmax, sigmaY=sigmay)


# Bilateral Filter: bilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp. However, it is very slow compared to most filters

# Parameters:

# d (Filter size): Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications, 
# and perhaps d=9 for offline applications that need heavy noise filtering.

# Sigma values: For simplicity, you can set the 2 sigma values to be the same. If they are small (<10), the filter will not have much effect, 
# whereas if they are large (> 150), they will have a very strong effect, making the image look "cartoonish" 
def bilateral_filter(img_gray: np.ndarray, d: int =5, sigmaColor: int =75, sigmaSpace: int =75) -> np.ndarray:
    return cv2.bilateralFilter(img_gray, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)


# Median Filter:

# Parameters:
# ksize: that must be a positive and odd number in range [1, 3, 5, ...]
# If ksize is 1 then output and input will be equal.
def median_filter(img_gray: np.ndarray, ksize: int =3) -> np.ndarray:
    return cv2.medianBlur(img_gray, ksize=ksize)


# Canny Edge Detection:

# Parameters: 
# threshold1: is better that be in the range [0, 255]
# threshold2: is better that be in the range [0, 255] and must be greater than threshold1.
def canny_filter(img_gray: np.ndarray, threshold1: int =100, threshold2: int =200) -> np.ndarray:
    return cv2.Canny(img_gray, threshold1=threshold1, threshold2=threshold2)
# End of Noise Removal Block



'''
# Test:
import matplotlib.pyplot as plt

path = '/home/shahin/Desktop/Majazi/ocr_project/testimage/'
img_name = '01.jpg'
img = cv2.imread(path + img_name, 0)
img_noise_removal = bilateral_filter(img)
print(np.unique(img == img_noise_removal))

plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(img_noise_removal, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()
'''