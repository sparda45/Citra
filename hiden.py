import cv2
import numpy as np
import pywt

def embed_text_into_coefficients(coefficients, binary_text):
    # Flatten the coefficients
    flattened = coefficients.flatten()

    # Ensure binary text length does not exceed the coefficient length
    binary_text = binary_text[:len(flattened)]

    # Embed text into coefficients
    for i, bit in enumerate(binary_text):
        flattened[i] = ((int(flattened[i]) & 254) | int(bit))

    # Reshape the coefficients
    return flattened.reshape(coefficients.shape)

def extract_text_from_coefficients(coefficients, length):
    # Flatten the coefficients and convert to uint8 to ensure compatibility with bitwise operation
    flattened = coefficients.astype(np.uint8).flatten()

    # Extract embedded text from coefficients
    extracted_text = ''
    for bit in flattened[:length]:
        extracted_text += str(bit & 1)

    # Ensure the extracted text length is a multiple of 8
    padding = len(extracted_text) % 8
    if padding != 0:
        extracted_text = extracted_text[:-padding]

    # Convert binary text to characters
    extracted_chars = [extracted_text[i:i+8] for i in range(0, len(extracted_text), 8)]
    extracted_bytes = [int(char, 2) for char in extracted_chars]
    extracted_text = bytes(extracted_bytes).decode('utf-8', 'ignore')

    return extracted_text

def encrypt_text_into_image(image_path, text, output_path):
    # Read the image without converting to grayscale
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not read the image.")
        return

    # Convert text to binary
    binary_text = ''.join(format(ord(char), '08b') for char in text)
    print("Original Text:", text)
    print("Binary Text:", binary_text)
    print("Binary Text Length:", len(binary_text))  # Print length of binary text

    # Convert image to YUV color space
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_img)

    # Perform 2-level discrete wavelet transform on the luminance (Y) channel
    coeffs = pywt.dwt2(y, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Embed text into wavelet coefficients
    embedded_coeffs = embed_text_into_coefficients(cA, binary_text)

    # Perform inverse 2-level discrete wavelet transform
    y_embedded = pywt.idwt2((embedded_coeffs, (cH, cV, cD)), 'haar')

    # Resize chrominance channels to match dimensions of the transformed luminance channel
    u_resized = cv2.resize(u, (y_embedded.shape[1], y_embedded.shape[0]), interpolation=cv2.INTER_LINEAR)
    v_resized = cv2.resize(v, (y_embedded.shape[1], y_embedded.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Convert Y channel back to uint8 (if needed)
    y_embedded = np.uint8(y_embedded)
    # Convert U and V channels back to uint8 (if needed)
    u_resized = np.uint8(u_resized)
    v_resized = np.uint8(v_resized)

    # Merge YUV channels back into the image
    encrypted_img = cv2.merge((y_embedded, u_resized, v_resized))

    # Convert image back to BGR color space
    encrypted_img = cv2.cvtColor(encrypted_img, cv2.COLOR_YUV2BGR)

    # Save the encrypted image
    cv2.imwrite(output_path, encrypted_img)
    print("Image encrypted successfully.")

def decrypt_text_from_image(image_path, text_length):
    # Read the image
    encrypted_img = cv2.imread(image_path)

    if encrypted_img is None:
        print("Error: Could not read the encrypted image.")
        return

    # Convert image to YUV color space
    yuv_img = cv2.cvtColor(encrypted_img, cv2.COLOR_BGR2YUV)
    y, _, _ = cv2.split(yuv_img)

    # Perform 2-level discrete wavelet transform on the luminance (Y) channel
    coeffs = pywt.dwt2(y, 'haar')
    cA, _ = coeffs

    # Extract embedded text from wavelet coefficients
    extracted_text = extract_text_from_coefficients(cA, text_length)

    return extracted_text

# Example usage:
image_path = 'messi10.png'
text_to_encrypt = "artinya apaan bang messi"
output_image_path = 'bismilah.png'

# Encrypt text into image
encrypt_text_into_image(image_path, text_to_encrypt, output_image_path)

# Decrypt text from image
decrypted_text = decrypt_text_from_image(output_image_path, len(text_to_encrypt) * 8)
print("Decrypted Text:", decrypted_text)
