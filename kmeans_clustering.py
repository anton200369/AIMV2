import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def kmeans_clustering(image_path, num_clusters):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}.")
        return None

    # Convert the image from RGB to Lab color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Get rid of the luminance channel
    image = image[:, :, 1:3]

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, image.shape[2]))
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8
    centers = np.uint8(centers)

    # Map each pixel to the centroid value
    clustered_image = centers[labels.flatten()]
    clustered_image = clustered_image.reshape(image.shape)

    # Display the clustered image
    cv2.namedWindow("Clustered Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Clustered Image", clustered_image)
    cv2.waitKey(0)

    # Calculate reconstruction error
    reconstruction_error = np.sqrt(np.mean((pixels - centers[labels.flatten()]) ** 2))

    # Print centroids and reconstruction error
    print(f"Image: {image_path}")
    print("Centroids:")
    for i, center in enumerate(centers):
        print(f"Centroid {i}: {center}")
    print(f"Reconstruction error: {reconstruction_error}")

    # Generate the concatenated centroid vector
    concatenated_centroids = generate_concatenated_centroid_vector(labels, centers)
    print(f"Concatenated Centroids: {concatenated_centroids}")

    return concatenated_centroids

def generate_concatenated_centroid_vector(labels, centers):
    """
    Generate a vector composed of the concatenation of the centroids ordered by cluster size.
    
    Parameters:
    labels (numpy.ndarray): The labels of each pixel.
    centers (numpy.ndarray): The centroids of the clusters.
    
    Returns:
    numpy.ndarray: The concatenated centroid vector.
    """
    # Calculate the size of each cluster
    cluster_sizes = np.bincount(labels.flatten())

    # Sort the clusters by size in descending order
    sorted_indices = np.argsort(-cluster_sizes)

    # Concatenate the centroids in the order specified
    concatenated_centroids = centers[sorted_indices].flatten()

    return concatenated_centroids

def process_folder(folder_path, num_clusters):
    concatenated_centroids_list = []
    image_paths = []

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            if not os.path.isfile(image_path):
                print(f"Warning: File {image_path} does not exist.")
                continue
            concatenated_centroids = kmeans_clustering(image_path, num_clusters)
            if concatenated_centroids is not None:
                concatenated_centroids_list.append(concatenated_centroids)
                image_paths.append(filename)

    if not concatenated_centroids_list:
        print("Error: No valid images found in the folder.")
        return
    
    # Calculate the distance matrix
    num_images = len(concatenated_centroids_list)
    distance_matrix = np.zeros((num_images, num_images))    
    for i in range(num_images):
        for j in range(num_images):
            distance_matrix[i, j] = np.linalg.norm(concatenated_centroids_list[i].astype(float) - concatenated_centroids_list[j].astype(float))

    # Plot the distance matrix  
    # Visualize the distance matrix
    plt.imshow(distance_matrix, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(image_paths)), image_paths, rotation=90, fontsize=8)
    plt.yticks(range(len(image_paths)), image_paths, fontsize=8)
    plt.title('Distance Matrix of Concatenated Centroids')
    plt.tight_layout()  # Adjust layout to make room for the labels
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python Demo_kmeans.py <folder_path> <num_clusters>")
        sys.exit(1)

    folder_path = sys.argv[1]
    num_clusters = int(sys.argv[2])

    process_folder(folder_path, num_clusters)