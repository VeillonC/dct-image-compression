"""
Compare several quantization matrices (low, JPEG, aggressive, extreme, max)
on the same image.
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

def compress_by_quantization(img, P, Q, n=8):
    nrows, ncols, _ = img.shape
    compressed = np.zeros_like(img)
    for c in range(3):
        for i in range(0, nrows, n):
            for j in range(0, ncols, n):
                block = img[i:i+n, j:j+n, c]
                D = np.dot(P, np.dot(block, P.T))
                compressed_block = np.round(D / Q)
                compressed[i:i+n, j:j+n, c] = compressed_block
    return compressed

def decompress_by_quantization(compressed, P, Q, n=8):
    nrows, ncols, _ = compressed.shape
    decompressed = np.zeros_like(compressed)
    for c in range(3):
        for i in range(0, nrows, n):
            for j in range(0, ncols, n):
                block = compressed[i:i+n, j:j+n, c]
                D_tilde = block * Q
                M_tilde = np.dot(P.T, np.dot(D_tilde, P))
                decompressed[i:i+n, j:j+n, c] = M_tilde
    decompressed = np.clip(decompressed + 128, 0, 255)
    return decompressed

def calculate_compression_rate(compressed, nrows, ncols):
    number = np.count_nonzero(compressed)
    return 100 - (round(number / (nrows * ncols * 3) * 100))

def display_images(original, decompressed_images, compression_rates, q_labels):
    plt.figure(figsize=(20, 10))

    # Original image
    plt.subplot(2, 3, 1)
    plt.title("Original image")
    plt.imshow(original + 128)

    # Decompressed images
    for idx, (decompressed, compression_rate) in enumerate(zip(decompressed_images, compression_rates)):
        plt.subplot(2, 3, idx + 2)
        plt.title(f"Compressed ({q_labels[idx]})")
        plt.imshow(decompressed)
        plt.text(0.5, -0.15, f"compression: {compression_rate}%", ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    image_path = "../images/waves.png"  # <- change this path to test another image
    img, nrows, ncols = initialize_image(image_path)

    P = create_dct_matrix()

    Q_classic = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 13, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    Q_aggressive = np.array([[32, 24, 20, 32, 48, 80, 102, 124],
                    [24, 24, 28, 38, 52, 116, 120, 110],
                    [28, 26, 32, 48, 80, 114, 138, 112],
                    [28, 34, 44, 58, 102, 174, 160, 124],
                    [36, 44, 74, 112, 136, 218, 206, 154],
                    [48, 70, 110, 128, 162, 208, 226, 184],
                    [98, 128, 156, 174, 206, 242, 240, 202],
                    [144, 184, 190, 196, 224, 200, 206, 198]
                    ])

    Q_low = np.array([[8, 6, 5, 8, 12, 20, 25, 30],
                    [6, 6, 7, 10, 13, 29, 30, 27],
                    [7, 6, 8, 12, 20, 28, 34, 28],
                    [7, 8, 11, 14, 25, 43, 40, 31],
                    [9, 11, 19, 28, 34, 54, 51, 38],
                    [12, 18, 28, 32, 41, 52, 56, 46],
                    [24, 32, 39, 43, 51, 60, 60, 50],
                    [36, 46, 48, 49, 56, 50, 51, 49]
                    ])

    Q_extreme = np.array([
                    [80, 80, 80, 80, 80, 80, 80, 80],
                    [80, 90, 90, 90, 90, 90, 90, 90],
                    [80, 90, 100, 100, 100, 100, 100, 100],
                    [80, 90, 100, 110, 110, 110, 110, 110],
                    [80, 90, 100, 110, 120, 120, 120, 120],
                    [80, 90, 100, 110, 120, 130, 130, 130],
                    [80, 90, 100, 110, 120, 130, 140, 140],
                    [80, 90, 100, 110, 120, 130, 140, 150]
                    ])

    Q_max = Q_extreme * 10

    list_Q = [Q_low, Q_classic, Q_aggressive, Q_extreme, Q_max]
    q_labels = ["low", "jpeg", "aggressive", "extreme", "max"]

    decompressed_images = []
    compression_rates = []

    for Q in list_Q:
        compressed = compress_by_quantization(img, P, Q)
        compression_rate = calculate_compression_rate(compressed, nrows, ncols)
        decompressed = decompress_by_quantization(compressed, P, Q)
        decompressed_images.append(decompressed)
        compression_rates.append(compression_rate)

    display_images(img, decompressed_images, compression_rates, q_labels)
