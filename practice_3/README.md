# Practice 3 - Image Clustering with KMeans

This script applies KMeans clustering to an input image and generates a clustered version of the image. 

## Installation
Install the required libraries:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

## Jupyter Notebook

This notebook provides visualizations and comparisons between a custom KMeans implementation and the `sklearn` KMeans. Follow the steps below to see how each implementation performs under various conditions.

---

This version improves readability and flow, ensuring that it’s clear the notebook offers step-by-step comparisons of the two implementations.

## Create clustered_image with python script
Run the script using the following command:

```bash
python main.py <image_path> <num_clusters> [--algorithm sklean/custom]  [--result_img_path <result_img_path>] 
```

### Arguments:
- `<image_path>`: Path to the input image (e.g., `images/sample.jpg`).
- `<num_clusters>`: Number of clusters for KMeans (e.g., `10`).
- `--result_img_path`: (Optional) Path to save the resulting image. Defaults to `result.jpg`.
- `--algorithm`: (Optional) Path to save the resulting image. Defaults to `result.jpg`.
--method: Choose between sklearn (scikit-learn's KMeans) or custom implementation.


### Example:
```bash
python main.py odessa.jpeg 10 --result_img_path clustered_odessa.jpg --method sklearn
```

This command processes `odessa.jpeg` with 10 clusters and saves the clustered image as `clustered_odessa.jpg`.

## Output
- The resulting clustered image is saved to the specified path (`--result_img_path` or default `result.jpg`).

## Notes
- The `MAX_POINT` parameter in `config` determines the resizing scale for the input image to ensure efficient clustering.
- Ensure all required modules (`kmean_from_scratch`, `utils`, `config`) are properly implemented and imported.



