"""Color palette extraction from images."""
from __future__ import annotations

from PIL import Image
import numpy as np


def extract_palette(image: Image.Image, n_colors: int = 5) -> list[str]:
    """Extract dominant color palette using k-means on downsampled image."""
    # Downsample for speed
    img = image.copy()
    img.thumbnail((150, 150))
    pixels = np.array(img.convert("RGB")).reshape(-1, 3)
    
    # Simple k-means (no sklearn dependency needed for this)
    from numpy.random import default_rng
    rng = default_rng(42)
    
    # Initialize centroids randomly
    indices = rng.choice(len(pixels), n_colors, replace=False)
    centroids = pixels[indices].astype(float)
    
    for _ in range(10):  # 10 iterations is enough
        # Assign pixels to nearest centroid
        dists = np.sqrt(((pixels[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        labels = dists.argmin(axis=1)
        
        # Update centroids
        for i in range(n_colors):
            mask = labels == i
            if mask.sum() > 0:
                centroids[i] = pixels[mask].mean(axis=0)
    
    # Convert to hex
    palette = []
    for c in centroids:
        hex_color = "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))
        palette.append(hex_color)
    
    return palette
