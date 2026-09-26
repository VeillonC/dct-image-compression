"""
Compare several frequency-filtering cutoffs (compression by filter) on the
same image.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg

def initialize_image(image_path):
    img = mpimg.imread(image_path)
    if len(img.shape) == 2 or img.shape[2] < 3:
        raise ValueError("The image must have three color channels (RGB).")
    if img.dtype != np.uint8:
        img = (img*255)
    else :
        img = (img/np.max(img))*255
    img = np.array(img, dtype=int)
    nrows, ncols, _ = img.shape
    nrows -= nrows % 8
    ncols -= ncols % 8
    img = img[:nrows, :ncols, :]
    img = img - 128
    return img, nrows, ncols

def create_dct_matrix(n=8):
    P = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            if k == 0:
                P[k, i] = 1 / math.sqrt(n)
            else:
                P[k, i] = math.sqrt(2 / n) * math.cos((2 * i + 1) * k * math.pi / (2 * n))
    return P

def compress_by_filter(image, P, cutoff):
    compressed = np.zeros_like(image)
    nrows, ncols, _ = image.shape

    for c in range(3):  
        for i in range(0, nrows, 8):
            for j in range(0, ncols, 8):
                block = image[i:i+8, j:j+8, c]  
                D = np.dot(P, np.dot(block, P.T))  
                for m in range(8):
                    for n in range(8):
                        if m + n >= cutoff:
                            D[m, n] = 0
                compressed[i:i+8, j:j+8, c] = D  

    compression_rate = 100 - round(np.count_nonzero(compressed) / (nrows * ncols * 3) * 100)
    return compressed, compression_rate

def decompress_by_filter(compressed, P, n=8):
    nrows, ncols, _ = compressed.shape
    decompressed = np.zeros_like(compressed)

    for c in range(3):  
        for i in range(0, nrows, n):
            for j in range(0, ncols, n):
                block = compressed[i:i+n, j:j+n, c]  
                M_tilde = np.dot(P.T, np.dot(block, P))  
                decompressed[i:i+n, j:j+n, c] = M_tilde  

    decompressed = np.clip(decompressed + 128, 0, 255)  
    return decompressed


def display_images(original, decompressed_images, compression_rates, f_labels):
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.title("Original image")
    plt.imshow(original + 128)
    plt.axis('off')

    for idx, (decompressed, compression_rate) in enumerate(zip(decompressed_images, compression_rates)):
        plt.subplot(2, 3, idx + 2)
        plt.title(f"Cutoff {f_labels[idx]}")
        plt.imshow(decompressed.astype(np.uint8))
        plt.axis('off')
        plt.text(0.5, -0.1, f"Compression: {compression_rate}%", ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "../images/waves.png"  # <- change this path to test another image
    img, nrows, ncols = initialize_image(image_path)

    P = create_dct_matrix()

    list_f = [10,6,4,2,1]
    f_labels = ["10", "6", "4", "2", "1"]

    decompressed_images = []
    compression_rates = []

    for f in list_f:
        compressed, compression_rate = compress_by_filter(img, P, f)  # Unpack both values
        decompressed = decompress_by_filter(compressed, P)
        decompressed_images.append(decompressed)
        compression_rates.append(compression_rate)
    display_images(img, decompressed_images, compression_rates, f_labels)
