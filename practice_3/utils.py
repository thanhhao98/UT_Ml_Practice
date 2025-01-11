import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_new_shape(origin_height, origin_width, max_point):
    origin_point = origin_width*origin_height
    if origin_point <= max_point:
        return origin_width, origin_height
    ratio = math.sqrt(max_point/origin_point)
    return int(origin_height*ratio), int(origin_width*ratio)

def reconstruct(img, high, width):
    reconstructed_img = img.reshape(high, width, 3)
    reconstructed_img = (reconstructed_img*255).astype(np.uint8)
    return reconstructed_img

def visualize_img(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()

def visualize_comparison(img_custom, img_sklearn, n_cluster):
    """Visualize side-by-side comparison of custom KMeans and sklearn KMeans clustering results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with 2 subplots (side by side)
    axes[0].imshow(cv2.cvtColor(img_custom, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Custom KMeans (n_clusters={n_cluster})")
    axes[0].axis('off')  # Hide axes for a cleaner look
    
    axes[1].imshow(cv2.cvtColor(img_sklearn, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"sklearn KMeans (n_clusters={n_cluster})")
    axes[1].axis('off')  # Hide axes for a cleaner look
    
    plt.tight_layout()
    plt.show()