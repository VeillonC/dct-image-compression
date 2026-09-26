"""
Compare several frequency filters with a fixed quantization matrix Q.
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

def compress_by_filter_and_quantization(img, P, Q, cutoff, n=8):
    nrows, ncols, _ = img.shape
    compressed = np.zeros_like(img)
    for c in range(3):
        for i in range(0, nrows, n):
            for j in range(0, ncols, n):
                block = img[i:i+n, j:j+n, c]
                D = np.dot(P, np.dot(block, P.T))
                temp = np.zeros((n, n))
                for m in range(min(cutoff, n)):
                    for l in range(min(cutoff, n) - m):
                        temp[m, l] = D[m, l]
                compressed_block = np.round(temp / Q)
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

def display_images(original, decompressed_images, compression_rates, q_labels, cutoff):
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.title("Original image")
    plt.imshow(original + 128)

    # Decompressed images
    for idx, (decompressed, compression_rate) in enumerate(zip(decompressed_images, compression_rates)):
        plt.subplot(2, 3, idx + 2)
        plt.title(f"Compressed ({q_labels[idx]}, cutoff: {cutoff})")
        plt.imshow(decompressed)
        # Add the compression rate below the image
        plt.text(0.5, -0.15, f"compression: {compression_rate}%", ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "../images/waves.png"  # <- change this path to test another image
    img, nrows, ncols = initialize_image(image_path)

    P = create_dct_matrix()
    cutoff = 6

    Q_low = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 13, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    Q_classic = Q_low * 4

    Q_aggressive = Q_low * 8

    Q_extreme = Q_low * 15

    Q_max = Q_low * 40

    list_q = [Q_low, Q_classic, Q_aggressive, Q_extreme, Q_max]
    q_labels = ["Q low", "Q x4", "Q x8", "Q x15", "Q x40"]

    decompressed_images = []
    compression_rates = []

    for Q in list_q:
        before = np.count_nonzero(img)
        compressed = compress_by_filter_and_quantization(img, P, Q, cutoff)
        after = np.count_nonzero(compressed)
        difference_before_after = round((before-after)/(nrows*ncols))*100
        compression_rate = calculate_compression_rate(compressed, nrows, ncols)
        decompressed = decompress_by_quantization(compressed, P, Q)
        decompressed_images.append(decompressed)
        compression_rates.append(compression_rate)
        print("non-zero coefficients before:", before, "\n", "after:", after, "\n")

    display_images(img, decompressed_images, compression_rates, q_labels, cutoff)
