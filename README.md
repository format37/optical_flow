# Optical Flow Analysis for Fluid Dynamics

This project implements Lucas-Kanade optical flow algorithm to track fluid motion patterns in video sequences. It's particularly useful for analyzing convection currents, eddy formations, and other fluid dynamic phenomena.

## Demo
[![Convection Currents Analysis](https://img.youtube.com/vi/XtssdjG79qs/0.jpg)](https://www.youtube.com/watch?v=XtssdjG79qs)

Click the image above to watch the demo video.

## How It Works

The script uses computer vision techniques to track the movement of points in a video sequence:

1. **Grid Point Initialization**: 
   - Creates a uniform grid of points across the video frame
   - Points are spaced evenly in both x and y directions
   - Default configuration uses a 10x10 grid (100 tracking points)

2. **Lucas-Kanade Optical Flow**:
   - Implements the Lucas-Kanade method for optical flow calculation
   - Tracks point movements between consecutive frames
   - Uses pyramidal implementation for handling larger movements
   - Key parameters:
     - Window size: 15x15 pixels
     - Pyramid levels: 2
     - Termination criteria: 10 iterations or 0.03 epsilon

3. **Trajectory Visualization**:
   - Each point is assigned a unique color
   - Trajectories are drawn as continuous lines
   - Real-time visualization of point movements
   - Final output saved as MP4 video

## Mathematical Background

The Lucas-Kanade method is based on three main assumptions:
1. Brightness Constancy: The brightness of a pixel doesn't change between consecutive frames
2. Temporal Persistence: The motion of a surface patch changes slowly in time
3. Spatial Coherence: Neighboring points belong to the same surface and have similar motion

The algorithm solves the optical flow equation:
```
Ix*u + Iy*v = -It
```
where:
- Ix, Iy are image gradients
- It is temporal gradient
- u, v are the velocity components we want to compute

## Usage

```python
from optical_flow import track_grid_points

# Track points in video
trajectories = track_grid_points('path_to_video.mp4', grid_size=10)
```

## Requirements

- OpenCV (cv2)
- NumPy
- Python 3.6+

## Installation

```bash
pip install opencv-python numpy
```

## Output

The script generates:
1. Real-time visualization window
2. Output video file ('output.mp4') with tracked trajectories
3. Trajectory data saved as CSV file ('trajectories.csv') for further analysis
4. Trajectory analysis visualization ('assets/trajectories_with_circles.png')

## Trajectory Analysis

The project includes advanced trajectory analysis capabilities:

1. **Data Processing**:
   - Trajectories are saved to CSV format with point IDs and coordinates
   - Outlier removal using z-score thresholding
   - Robust handling of incomplete or noisy trajectories

2. **Circle Fitting**:
   - Fits circular paths to each point trajectory
   - Uses least squares optimization to find best-fit circles
   - Calculates center coordinates and radius for each trajectory
   - Determines median circle parameters across all trajectories

3. **Visualization**:
   - Plots all point trajectories with fitted circles
   - Highlights median circle and center point
   - Provides insights into overall fluid motion patterns
   - Output saved as 'trajectories_with_circles.png'

![Trajectory Analysis](assets/trajectories_with_circles.png)

## License

MIT License

## Contributing

Feel free to open issues and pull requests for improvements!