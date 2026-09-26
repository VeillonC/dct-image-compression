# Image Compression with the Discrete Cosine Transform (DCT)

A from-scratch implementation (pure NumPy, no compression library) of a JPEG-style image compression/decompression pipeline, based on the block-wise 8x8 DCT-2.

Academic group project (4 students), MAM3 — Polytech Nice Sophia, Applied Mathematics and Modeling engineering program.

## How it works

1. **Split** the image into 8x8 pixel blocks, per color channel (R, G, B).
2. **DCT-2** of each block to move into the frequency domain.
3. **Compression** of the coefficients, using two approaches compared in this project:
   - **quantization**: divide the coefficients by a matrix `Q` (more or less aggressive) and round;
   - **filtering**: zero out high-frequency coefficients beyond a threshold (cutoff).
4. **Decompression**: inverse DCT to reconstruct the image.
5. **Evaluation**: compression rate (share of coefficients set to zero) and visual quality of the reconstructed image.

The full report (`docs/report.pdf`, in French) details the formulas, the choice of quantization matrices, and the analysis of results (including on a noisy image).

## Repository structure

```
compression-image-dct/
├── src/
│   ├── algorithm.py                          # Shared functions: DCT, compression/decompression
│   ├── compare_quantization.py               # Compares several Q matrices
│   ├── compare_quantization_fixed_filter.py  # Compares Q matrices, fixed filter
│   ├── compare_filter.py                     # Compares several filter cutoffs
│   └── compare_filter_fixed_quantization.py  # Compares filters, fixed Q matrix
├── images/                                   # Test images (RGB)
├── assets/                                   # Visuals used in this README
└── docs/
    ├── report.pdf                            # Full report (French)
    └── slides.pdf                            # Presentation slides (French)
```

## Installation

```bash
git clone https://github.com/<your-username>/compression-image-dct.git
cd compression-image-dct
pip install -r requirements.txt
```

## Usage

Each script in `src/` is self-contained and should be run from that folder:

```bash
cd src
python compare_quantization.py
```

To test another image, edit the `image_path` variable at the top of the `if __name__ == "__main__":` block of each script (sample images are provided in `images/`).

## Results

On the `waves.png` image (shown above): a compression rate of 96 to 98% is reached while preserving good visual quality. The full report (`docs/report.pdf`) covers results on more images, including a noisy one, with the associated error rates.

## Authors

Group project by Matthieu Keruzoret, Petru Piculescu, Camille Veillon and Zineb Ziad.
