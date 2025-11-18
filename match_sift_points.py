import cv2
import numpy as np
import argparse

def match_sift_points(image1_path, image2_path):

    # Read images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # verify they are not None
    if img1 is None or img2 is None:
        print("Error al cargar las imágenes.")
        return
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=50)

    # Read the image to compare to
    IDCardTemplate1 = img1
    IDCardTemplate2 = img2

    # Resize both templates to have 425x270 pixels
    IDCardTemplate1 = cv2.resize(IDCardTemplate1, (425, 270))
    IDCardTemplate2 = cv2.resize(IDCardTemplate2, (425, 270))

    # For visualization, create a  image that stacks the two card templates in two rows (inefficient)
    IDCardTemplate = np.vstack((IDCardTemplate1,  IDCardTemplate2))

    # Detect SIFT features in the ID card templates
    IDCard_keypoints1, IDCard_descriptors1 = sift.detectAndCompute(IDCardTemplate1, None)
    IDCard_keypoints2, IDCard_descriptors2 = sift.detectAndCompute(IDCardTemplate2, None)

    # Detect also SIFT features in the combined ID card templates for visualization (inefficient)
    IDCard_keypoints, IDCard_descriptors = sift.detectAndCompute(IDCardTemplate, None)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read the next frame
        success, frame = cap.read()
        frame = cv2.resize(frame, (960, 540)) # Resize the frame
        if not success:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect SIFT features in the frames
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Match the features using FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches1 = flann.knnMatch(IDCard_descriptors1, descriptors, k=2)
        matches2 = flann.knnMatch(IDCard_descriptors2, descriptors, k=2)
        matches = flann.knnMatch(IDCard_descriptors, descriptors, k=2) #(inefficient) only for visualization

        # Filter matches using the Lowe's ratio test: take only the good matches (those with a feature distance that is clearly smaller than the second best match)
        good_matches1 = [m for m, n in matches1 if m.distance < 0.5 * n.distance]
        good_matches2 = [m for m, n in matches2 if m.distance < 0.5 * n.distance]

        # Compare the number of good matches
        template_number = 1 if len(good_matches1) > len(good_matches2) else 2

        # This finbal part is only for demonstration purposes
        # Overlay the template number on the video frame if it has enough matches (in this case 5 is dependent on the thresholds set for SIFT)

        if max(len(good_matches1), len(good_matches2)) > 5:
            cv2.putText(frame, str(template_number), (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Need to draw only good matches in the visualization, so create a mask 
        good_matches = [[0, 0] for i in range(len(matches))] 
        
        # Good matches 
        for i, (m, n) in enumerate(matches): 
            if m.distance < 0.5*n.distance: 
                good_matches[i] = [1, 0] 

        # Draw matches 
        match_frame = cv2.drawMatchesKnn(IDCardTemplate, 
                             IDCard_keypoints, 
                             frame, 
                             keypoints, 
                             matches, 
                             outImg=None, 
                             #matchColor=(0, 155, 0), 
                             #singlePointColor=(0, 255, 255), 
                             matchesMask=good_matches, 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                             ) 
        
        # Display the frame with matches
        cv2.imshow("Matches", match_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compare two images with the webcam flow using SIFT.')
    parser.add_argument('image1', type=str, help='path of the image 1')
    parser.add_argument('image2', type=str, help='path of the image 2')
    args = parser.parse_args()

    match_sift_points(args.image1, args.image2)
