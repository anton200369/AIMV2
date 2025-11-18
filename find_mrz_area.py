import cv2
import numpy as np
import sys

def find_mrz_area(image):
    """
    Find the area where the MRZ is located in an ID card image.
    
    Parameters:
    image (numpy.ndarray): The input ID card image.
    
    Returns:
    tuple: The bounding box (x, y, w, h) of the MRZ area.
    """

    # initialize a rectangular and square structuring kernel (this size is dependent on the ID-Card size)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 31))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve contour detection
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imshow('Gray', gray)

    # Apply blackhat morphological operation to enhance the text
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    cv2.imshow('Blackhat', blackhat)

    # apply a closing operation using the rectangular kernel to close
	# gaps in between letters -- then apply Otsu's thresholding method
    blackhat_closed = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, rectKernel)
    cv2.imshow('blackhat_closed', blackhat_closed)
    thresh = cv2.threshold(blackhat_closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('Thresh', thresh)

    # perform another closing operation, this time using the square
	# kernel to close gaps between lines of the MRZ
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    cv2.imshow('Thresh_closed', thresh)

    # Convert the thresholded image to a color image to plot the contours in color
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
	# find contours in the thresholded image and sort them by their
	# size
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    # Handle different versions of OpenCV
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Sort contours by area in descending order    
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Initialize ROI as None
    roi = None

	# loop over the contours
    for i, c in enumerate(cnts):
		# compute the bounding box of the contour and use the contour to
		# compute the aspect ratio and coverage ratio of the bounding box
		# width to the width of the image
        (x, y, w, h) = cv2.boundingRect(c)
         # Draw the bounding box on the color thresholded image
        color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for the first (largest) contour, red for others
        cv2.rectangle(thresh_color, (x, y), (x + w, y + h), color, 2)
        cv2.imshow('Thresholded Image with Contours', thresh_color)

        ar = w / float(h)
        crWidth = w / float(gray.shape[1])
		# check to see if the aspect ratio and coverage width are within
		# acceptable criteria
        if ar > 4 and crWidth > 0.75:
			# pad the bounding box to have some space for later reading
            pad = 5
            x = x - pad
            y = y - pad
            w = w + (pad * 2)
            h = h + (pad * 2)

			# extract the ROI from the image and draw a bounding box
			# surrounding the MRZ
            roi = image[y:y + h, x:x + w].copy()

            break

    return (x, y, w, h), roi if roi is not None else (None, None)

def main():
    if len(sys.argv) != 2:
        print("Usage: python Crop_MRZ.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}.")
        sys.exit(1)

    # Scale the image to a constant size (e.g., width = 1000 pixels)
    scale_width = 1000
    scale_ratio = scale_width / image.shape[1]
    scaled_image = cv2.resize(image, (scale_width, int(image.shape[0] * scale_ratio)))

    # Find the MRZ area in the scaled image
    mrz_area, roi = find_mrz_area(scaled_image)
    if mrz_area is not None:
        x, y, w, h = mrz_area
        cv2.rectangle(scaled_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('MRZ Area', scaled_image)
        cv2.imshow('MRZ ROI', cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("MRZ area not found.")

if __name__ == "__main__":
    main()