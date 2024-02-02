import cv2
import pywt
import numpy as np

# Load the modified image
modified_img = cv2.imread('modified_image.png')

# Apply the wavelet transform
coefficients = pywt.dwt2(modified_img, 'haar')

# Extract the LSBs from the coefficients
extracted_bits = []
for channel in coefficients:
    for subband in channel:
        subband_bits = []
        for row in subband:
            for pixel in row:
                # Check if the pixel is not empty and has the expected shape
                if pixel.size > 0 and pixel.shape == (3,):  # Assuming it's a 3-channel pixel
                    # Ensure pixel values are integers
                    pixel = np.array(pixel, dtype=np.uint8)
                    # Extract the LSB from each channel of the pixel
                    pixel_bits = [int(p) & 1 for p in pixel]
                    # Convert the LSBs to a single value (byte)
                    subband_bits.append(np.packbits(np.array(pixel_bits)))
        extracted_bits.append(subband_bits)

# Convert the extracted bits to bytes
extracted_bytes = np.concatenate(extracted_bits)

# Write the extracted message to a text file
with open('extracted_message.txt', 'wb') as file:
    file.write(extracted_bytes)
