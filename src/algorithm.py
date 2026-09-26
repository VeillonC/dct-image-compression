"""
Core library: block-wise 8x8 DCT-2, compression/decompression by quantization
and by frequency filtering.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg

def initialize_image(image_path):
    img = mpimg.imread(image_path)
    if len(img.shape) == 2 or img.shape[2] < 3:
        raise ValueError("The image must have three color channels (RGB).")
    img = (img * 255) if img.dtype != np.uint8 else (img / np.max(img)) * 255
    img = np.array(img, dtype=int)
    nrows, ncols, _ = img.shape
    nrows -= nrows % 8
    ncols -= ncols % 8
    img = img[:nrows, :ncols, :] - 128
    return img, nrows, ncols

def create_dct_matrix(n=8):
    P = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            P[k, i] = (1 / math.sqrt(n) if k == 0 else 
                       math.sqrt(2 / n) * math.cos((2 * i + 1) * k * math.pi / (2 * n)))
    return P

def compress_by_quantization(image, P, Q):
    compressed = np.zeros_like(image)
    nrows, ncols, _ = image.shape
    
    for c in range(3): 
        for i in range(0, nrows, 8):
            for j in range(0, ncols, 8):
                block = image[i:i+8, j:j+8, c]  
                D = np.dot(P, np.dot(block, P.T))
                compressed_block = np.round(D / Q)  
                compressed[i:i+8, j:j+8, c] = compressed_block

    compression_rate = 100 - round(np.count_nonzero(compressed) / (nrows * ncols * 3) * 100)
    return compressed, compression_rate


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


if __name__ == "__main__":
    image_path = "../images/waves.png"  # <- change this path to test another image
    image, nrows, ncols = initialize_image(image_path)

    P = create_dct_matrix()

    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 13, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    cutoff = 6

    compressed_q, rate = compress_by_quantization(image, P, Q)
    decompressed_q = decompress_by_quantization(compressed_q, P, Q)

    compressed_f, rate_f = compress_by_filter(image, P, cutoff)
    decompressed_f = decompress_by_filter(compressed_f, P)


    plt.subplot(1, 3, 1)
    plt.imshow((image + 128).clip(0, 255))
    plt.title('Original image')

    plt.subplot(1, 3, 2)
    plt.imshow(decompressed_q)
    plt.title(f'Quantization decompression\ncompression: {rate}%')

    plt.subplot(1, 3, 3)
    plt.imshow(decompressed_f)
    plt.title(f'Filter decompression, cutoff={cutoff}\ncompression: {rate_f}%')

    plt.show()
