import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import matplotlib.pyplot as plt

def remove_outliers(data, columns, z_threshold=2):
    print(f"Using z-score threshold: {z_threshold}")
    print(f"Length before removing outliers: {len(data)}")
    
    for column in columns:
        mean = data[column].mean()
        std = data[column].std()
        z_scores = abs((data[column] - mean) / std)
        data = data[z_scores < z_threshold]
        print(f"Length after removing {column} outliers: {len(data)}")
    
    return data

def fit_circle(x, y):
    """
    Fit a circle to the given x,y points by minimizing the difference
    between the radius of each point and the mean radius.
    Returns (xc, yc, R).
    """
    # Initial guess for center (using centroid of the points)
    x_m = np.mean(x)
    y_m = np.mean(y)
    
    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    # Residuals function: difference between each distance and the mean distance
    def f_center(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = (x_m, y_m)
    # Solve for center (xc, yc)
    res_lsq = least_squares(f_center, center_estimate)
    xc, yc = res_lsq.x
    
    # Compute radius as the mean distance to the center
    R = np.mean(calc_R(xc, yc))
    return xc, yc, R

# ----------------------------------------------------------------------
# Plot circles for all points
df = pd.read_csv('trajectories.csv')
point_ids = df['point_id'].unique()

plt.figure(figsize=(15, 10))

# Lists to store circle parameters
centers_x = []
centers_y = []
radii = []

for point_id in point_ids:
    # Filter data for current point
    point = df[df['point_id'] == point_id]
    
    # Skip if too few points
    if len(point) < 3:
        continue
        
    # Apply outlier removal
    point_cleaned = remove_outliers(point, ['x', 'y'])
    
    # Skip if too few points after cleaning
    if len(point_cleaned) < 3:
        continue
    
    # Extract coordinates
    x_vals = point_cleaned['x'].values
    y_vals = point_cleaned['y'].values
    
    try:
        # Fit circle
        xc, yc, R = fit_circle(x_vals, y_vals)
        centers_x.append(xc)
        centers_y.append(yc)
        radii.append(R)
        
        # Plot trajectory
        plt.plot(x_vals, y_vals, '.-', alpha=0.3, markersize=2)
        
    except:
        print(f"Could not fit circle for point {point_id}")
        continue

# Calculate median circle parameters
median_x = np.median(centers_x)
median_y = np.median(centers_y) 
median_r = np.median(radii)

# Plot median circle
theta_fit = np.linspace(0, 2*np.pi, 100)
circle_x = median_x + median_r * np.cos(theta_fit)
circle_y = median_y + median_r * np.sin(theta_fit)
plt.plot(circle_x, circle_y, 'r-', linewidth=2, label='Median Circle')
plt.plot(median_x, median_y, 'r*', markersize=10, label='Median Center')

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate') 
plt.title('Trajectories with Fitted Circles for All Points')
plt.grid(True)
plt.gca().invert_yaxis()

# Save to assets
plt.savefig('assets/trajectories_with_circles.png')

plt.show()
