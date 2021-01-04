
import math

import numpy
from numpy import where, median, mean
from skimage import img_as_float
from skimage.io import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def mse(i1, i2):
    if i1.shape[0] != i2.shape[0] or i1.shape[1] != i2.shape[1]:
        raise ValueError("dimension mismatch")
    mse = mean(pow(i1 - i2, 2))
    return mse


image = imread('parrots.jpg')
as_float = img_as_float(image)
img_height = as_float.shape[0]
img_width = as_float.shape[1]

shaped_to_fit = as_float.reshape((img_height * img_width, 3))

plt.imshow(as_float)
plt.title('original')
plt.show()

for approx_method in [mean, median]:
    print("method", approx_method)
    for cluster_count in range(1, 21):
        kmeans = KMeans(init='k-means++', random_state=241, n_clusters=cluster_count)
        kmeans.fit(shaped_to_fit)
        clusters = kmeans.predict(shaped_to_fit)

        recolored = numpy.empty([shaped_to_fit.shape[0], shaped_to_fit.shape[1]])
        for cn in range(0, cluster_count):
            cluster_indexes = where(clusters == cn)
            cluster_values = shaped_to_fit[cluster_indexes]
            for m in range(shaped_to_fit.shape[1]):
                #approx_val = kmeans.cluster_centers_[cn, m]
                approx_val = approx_method(cluster_values[:, m])
                recolored[cluster_indexes[0], m] = approx_val

        psnr = 10 * math.log10(1.0 / mse(shaped_to_fit, recolored))
        print("psnr", cluster_count, ":", psnr)

        plt.imshow(recolored.reshape((img_height, img_width, 3)))
        plt.title('colors: ' + str(cluster_count))
        plt.show()


