import cv2
import numpy as np
import argparse
from sklearn.cluster import KMeans
from kmean_from_scratch import My_KMeans
from utils import (
    get_new_shape,
    reconstruct,
    visualize_img
)
from config import MAX_POINT

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Image clustering with KMeans.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("num_clusters", type=int, help="Number of clusters for KMeans.")
    parser.add_argument("--result_img_path", type=str, default="result.jpg", help="Path to save the resulting image.")
    parser.add_argument("--method", type=str, choices=["sklearn", "custom"], required=True, help="Choose the KMeans implementation: 'sklearn' or 'custom'.")

    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Unable to load image at {args.image_path}")
        return

    high, width = img.shape[0], img.shape[1]
    new_high, new_width = get_new_shape(high, width, 1000)

    # Resize image
    resized_img = cv2.resize(img, (new_width, new_high))
    img_float = resized_img / 255.0
    flatten_img_float = img_float.reshape(-1, 3)

    # Apply KMeans
    if args.method == "sklearn":
        kmeans = KMeans(n_clusters=args.num_clusters)
    else:
        kmeans = My_KMeans(n_clusters=args.num_clusters)
    kmeans.fit(flatten_img_float)
    img_hat = kmeans.cluster_centers_[kmeans.labels_]

    # Reconstruct image
    img_hat = reconstruct(img_hat, new_high, new_width)
    origin_img_hat = cv2.resize(img_hat, (width, high))

    # Save and visualize result
    cv2.imwrite(args.result_img_path, origin_img_hat)
    print(f"Resulting image saved to {args.result_img_path}")

if __name__ == "__main__":
    main()
