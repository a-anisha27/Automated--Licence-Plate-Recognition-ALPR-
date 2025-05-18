import cv2
import easyocr
import matplotlib.pyplot as plt
import imutils
import numpy as np

def detect_license_plate(image_path):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en']) # Use 'en' for English, add more languages if needed

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found")
        return

    # Resize image for better processing (optional)
    img = imutils.resize(img, width=800)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection using Canny
    edged = cv2.Canny(gray, 170, 200)

    # Find contours
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] # Get top 10 contours

    plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # Assume license plate is a rectangle with 4 vertices
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is None:
        print("No license plate detected")
        return

    # Draw the license plate contour
    cv2.drawContours(img, [plate_contour], -1, (0, 255, 0), 3)

    # Extract the license plate region
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    plate = cv2.bitwise_and(img, img, mask=mask)

    # Crop the license plate
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped_plate = gray[topx:bottomx+1, topy:bottomy+1]

    # Use EasyOCR to read text from the cropped plate
    results = reader.readtext(cropped_plate)

    # Extract and clean the license plate text
    plate_text = ""
    if results:
        plate_text = max(results, key=lambda x: x[2])[1] # Select the result with highest confidence
        plate_text = plate_text.replace(" ", "").upper() # Format text

    # Display results
    plt.figure(figsize=(12, 8))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(132), plt.imshow(cropped_plate, cmap='gray'), plt.title('Cropped Plate')
    plt.subplot(133), plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)), plt.title('Detected Plate')
    plt.show()

    return plate_text

# Example usage
if __name__ == "__main__":
    image_path = "car_image.jpg" # Replace with your image path
    text = detect_license_plate(image_path)
    print(f"Detected License Plate: {text}")
