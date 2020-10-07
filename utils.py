import numpy as np 
import matplotlib.pyplot as plt
import skimage
from skimage.feature import hog

def dist_weighted_entropy(img, W_radii=None):
    """
    Computes entropy by using inverse square distance weighted histogram
    W_radii serves as a cutoff on the distances which are considered, mainly for performance purposes
    This is nifty, but cubic. Alternatively usings pts where img==v and computing pairwise distance is probably a faster but still cubic alg
    """
    img = np.trunc(img)
    rows, cols = img.shape
    E = np.zeros((rows, cols))
    #Inverse squared distances
    W = np.array([])
    W_rows, W_cols = -1,-1
    if W_radii:
        W_rows, W_cols = W_radii
    else:
        W_rows, W_cols = rows, cols
    W = np.array([[1/max((i**2+j**2),1) for j in range(-W_cols,W_cols+1)] 
                      for i in range(-W_rows,W_rows+1)])
    def W_viewer(x,y):
        xmin = 0 if x > W_rows else W_rows - x
        xmax = 2*W_rows+1 if x + W_rows < rows else W_rows + rows - x 
        ymin = 0 if y > W_cols else W_cols - y
        ymax = 2*W_cols+1 if y + W_cols < cols else W_cols + cols - y
        return W[xmin:xmax,ymin:ymax]
    for v in np.arange(256):
        mask = (img==v)
        for x,y in np.transpose(np.nonzero(mask)):
            mask_view = mask[max(x-W_rows,0):x+W_rows+1,max(y-W_cols,0):y+W_cols+1]
            W_view = W_viewer(x,y)
            #About 10% faster than sum of the elementwise product
            prob = np.einsum("ij,ij->", mask_view, W_view) / np.sum(W_view)
            E[x,y] = -prob * np.log(prob)
    return E
	
def rgb2gray(img):
    #Usual RGB conversion, but skimage normalizes to [0,1) which makes the look up table harder
    return np.clip(np.dot(img[:,:,0:3],np.array([0.2125, 0.7154, 0.0721])),0,255)

def global_entropy(img):
    img_gray = rgb2gray(img)
    histo, _ = np.histogram(img_gray, bins=np.arange(256), density=True)
    H = -histo * np.log(histo)
    #Must clip exactly 255 out - the last bin is inclusive but array indices are not
    return H[np.clip(img_gray,0,254.999).astype(int)]

def global_hue_histo(img, n_bins):
    hsv = skimage.colors.rgb2hsv(img)
    bins = np.linspace(0, 255.0, n_bins)
    return np.histogram(hsv[:,:], bins=hue_bins, density=True)

def local_histo(data, bins, cell_size):
    x_cells, y_cells = data.shape[0] // cell_size[0], data.shape[1] // cell_size[1]
    H = np.zeros((x_cells, y_cells, bins.shape[0]-1))
    for i in range(x_cells):
        for j in range(y_cells):
            x, y = i*cell_size[0], j*cell_size[1]
            H[i,j], _ = np.histogram(data[x:x+cell_size[0],y:y+cell_size[1]], bins=bins)
            H[i,j] = H[i,j] / np.sum(H[i,j])
    return H
	
def features(img, n_hog_bins = 8, n_hue_bins=8, n_entropy_bins=8, W_radii=(3,3), cell_size=(8,8)):
    """
	img: RGB Image
	n_hog_bins: Number of HOG bins for orientations
	n_hue_bins: Number of hue bins for cell histograms
	n_entropy_bins: Number of entropy bins for cell histograms
	W_radii : A tuple of 2 integers x,y which provide maximum radii for the weighted histogram calculations
	cell_size : A tuple of 2 integers which are the rows and columns of the cells. This assumes the image is evenly divisble into these cells
    Returns: Flattened feature vector of the image:
	histogram of oriented gradients,
	hue histograms on a per cell basis,
    and entropy histogram on a per cell basis.
    """
    hue = skimage.color.rgb2hsv(img)[:,:,0]
    hue_bins = np.linspace(0, 1.0, n_hue_bins+1)
    hue_histo = local_histo(hue, hue_bins, cell_size)
    img_gray = rgb2gray(img)
    img_hog = hog(img_gray, orientations=n_hog_bins, pixels_per_cell=cell_size, 
                  cells_per_block=(1,1), block_norm='L2', multichannel=False)
    img_ent = dist_weighted_entropy(img_gray, W_radii=W_radii)
    ent_bins = np.linspace(0, 1, n_entropy_bins+1)
    ent_histo = local_histo(img_ent, ent_bins, cell_size)
    return np.concatenate((img_hog, hue_histo, ent_histo),axis=None)
	
def download_cifar(save=False):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    if save:
        np.save("./cifar_data/x_train", x_train)
        np.save("./cifar_data/y_train", y_train)
        np.save("./cifar_data/x_test", x_test)
        np.save("./cifar_data/y_test", y_test)
		
def load_cifar(raw_data=True):
	if raw_data:
		x_train = np.load("./cifar_data/x_train.npy", allow_pickle=True)
		y_train = np.load("./cifar_data/y_train.npy", allow_pickle=True).astype("float32")
		x_test = np.load("./cifar_data/x_test.npy", allow_pickle=True)
		y_test = np.load("./cifar_data/y_test.npy", allow_pickle=True).astype("float32")
		return x_train, y_train, x_test, y_test
	else:
		#These were made with: features(x, n_hog_bins = 8, n_hue_bins=8, n_entropy_bins=8, cell_size=(8,8))
		x_train = np.load("./cifar_data/x_train_features.npy", allow_pickle=True)
		y_train = np.load("./cifar_data/y_train.npy", allow_pickle=True).astype("float32")
		x_test = np.load("./cifar_data/x_test_features.npy", allow_pickle=True)
		y_test = np.load("./cifar_data/y_test.npy", allow_pickle=True).astype("float32")
		return x_train, y_train, x_test, y_test
    
    