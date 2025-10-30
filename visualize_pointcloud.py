import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the point cloud and colors from npz file
data = np.load('output/q3/points3d.npz')
points = data['points_3d']
colors = data['colors']

# Use only the first N colors to match the number of points
colors = colors[:len(points)]

# Convert from BGR (OpenCV format) to RGB and normalize to [0, 1]
colors_rgb = colors[:, ::-1] / 255.0

# Convert from homogeneous to Euclidean coordinates
# Divide x, y, z by the fourth coordinate (w)
x = points[:, 0] / points[:, 3]
y = points[:, 1] / points[:, 3]
z = points[:, 2] / points[:, 3]

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot points with their actual colors
ax.scatter(x, y, z, c=colors_rgb, marker='o', s=20, alpha=0.8)

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Colored 3D Point Cloud (Normalized from Projective Space)')

# Make it interactive
plt.tight_layout()
plt.show()
