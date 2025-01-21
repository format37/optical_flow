import cv2 as cv
import numpy as np
import pandas as pd
import os

def create_grid_points(frame, grid_size):
    """
    Create a grid of points on the frame
    
    Parameters:
    frame: video frame
    grid_size: int, number of points in each dimension (grid_size x grid_size total points)
    """
    height, width = frame.shape[:2]
    
    # Calculate spacing between points
    x_spacing = width // (grid_size + 1)
    y_spacing = height // (grid_size + 1)
    
    # Create grid points
    points = []
    for y in range(1, grid_size + 1):
        for x in range(1, grid_size + 1):
            points.append([x * x_spacing, y * y_spacing])
            
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)

def track_grid_points(videoPath, grid_size=10, frame_limit=100, marker_id=0):
    """
    Track a fixed grid of points using Lucas-Kanade optical flow (Option A approach).

    Unlike the original version, this function keeps a fixed-size array for
    all points (and a "active mask") so that each point's index never changes,
    even if the point is lost. This way we can reliably track a specific
    point ID (such as marker_id=53) across the entire video without the ID
    suddenly referring to a different real-world point.

    Parameters:
    -----------
    videoPath: str
        Path to the video file.
    grid_size: int
        Number of points in each dimension (creates grid_size x grid_size points).
    frame_limit: int
        Maximum number of frames to process (default: 100, use None for no limit).
    marker_id: int
        The index of the point to highlight in every frame (e.g., 53).

    Returns:
    --------
    trajectories: list of lists
        The trajectory for each point, where trajectories[i] is a list of (x, y) positions.
    df: pd.DataFrame
        A DataFrame of all tracked positions with columns:
        [point_id, frame, x, y].
    """
    cap = cv.VideoCapture(videoPath)
    if not cap.isOpened():
        print(f"Error: Could not open video at {videoPath}")
        return None, None

    # Read the first frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read first frame.")
        cap.release()
        return None, None

    # Create an initial set of grid points
    p0_original = create_grid_points(frame, grid_size)  # shape: (N, 1, 2)
    total_points = len(p0_original)
    print(f"Tracking {total_points} points in a {grid_size}x{grid_size} grid")

    # Convert the first frame to grayscale
    old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # We use a separate array "positions" to store the current coordinate for each point
    # throughout the video, even if it is "lost".
    positions = np.copy(p0_original)  # shape: (N, 1, 2)

    # Active mask: True if the point is still being tracked, False if lost
    active = [True] * total_points

    # Create colors for each point
    colors = np.random.randint(0, 255, (total_points, 3)).tolist()

    # Store trajectories for all points
    trajectories = [[] for _ in range(total_points)]
    frame_numbers = [[] for _ in range(total_points)]

    # Initialize each point's trajectory with the starting position
    for i, pt in enumerate(positions):
        x, y = pt[0]
        trajectories[i].append((int(x), int(y)))
        frame_numbers[i].append(0)

    # Prepare a mask for drawing
    mask = np.zeros_like(frame)

    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Video writer setup
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS)) if cap.get(cv.CAP_PROP_FPS) > 0 else 30

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    current_frame = 0

    try:
        while True:
            if frame_limit is not None and current_frame >= frame_limit:
                print(f"Reached frame limit of {frame_limit}")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_frame += 1
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Build arrays of only active points (so we can pass them to calcOpticalFlowPyrLK)
            active_ids = [i for i, a in enumerate(active) if a]
            if not active_ids:
                print("All points have been lost.")
                break

            # Build p0 array of only the active points
            p0_active = positions[active_ids]  # shape: (num_active, 1, 2)

            # Calculate optical flow for these active points
            p1, status, _err = cv.calcOpticalFlowPyrLK(
                old_gray,
                frame_gray,
                p0_active,
                None,
                **lk_params
            )

            # Update the big positions array
            for i, idx in enumerate(active_ids):
                if status[i][0] == 1:
                    # This point is successfully tracked
                    new_x, new_y = p1[i][0]
                    old_x, old_y = positions[idx][0]

                    # Draw the line of motion between old and new positions
                    mask = cv.line(mask, (int(old_x), int(old_y)), (int(new_x), int(new_y)), colors[idx], 2)
                    frame = cv.circle(frame, (int(new_x), int(new_y)), 3, colors[idx], -1)

                    # Optionally draw 'X' if this is the marker_id
                    if idx == marker_id:
                        size = 5
                        cv.line(frame,
                                (int(new_x) - size, int(new_y) - size),
                                (int(new_x) + size, int(new_y) + size),
                                (255, 255, 255), 2)
                        cv.line(frame,
                                (int(new_x) - size, int(new_y) + size),
                                (int(new_x) + size, int(new_y) - size),
                                (255, 255, 255), 2)

                    # Update main positions array with the new location
                    positions[idx][0] = (new_x, new_y)

                    # Store in trajectories
                    trajectories[idx].append((int(new_x), int(new_y)))
                    frame_numbers[idx].append(current_frame)
                else:
                    # Mark this point as lost
                    active[idx] = False

            # Combine frame + the accumulated mask
            img = cv.add(frame, mask)
            out.write(img)
            cv.imshow('Grid Point Tracking (Option A)', img)

            old_gray = frame_gray.copy()

            key = cv.waitKey(30) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print(f"Error during tracking: {str(e)}")

    finally:
        cap.release()
        out.release()
        cv.destroyAllWindows()

        # Build a DataFrame from trajectories
        data = []
        for point_idx in range(total_points):
            for f_idx, (x, y) in enumerate(trajectories[point_idx]):
                data.append({
                    'point_id': point_idx,
                    'frame': frame_numbers[point_idx][f_idx],
                    'x': x,
                    'y': y
                })
        df = pd.DataFrame(data)
        csv_path = 'trajectories_fixed.csv'
        df.to_csv(csv_path, index=False)
        print(f"Trajectory data saved to {csv_path}")

    print("Video saved to output.mp4")
    return trajectories, df

if __name__ == '__main__':
    videoPath = 'assets/Convection Currents in Infrared.mp4'
    if not os.path.exists(videoPath):
        print(f"Error: Video file not found at {videoPath}")
        exit(1)
    # Track points in a 10x10 grid, process up to 10 frames
    trajectories, df = track_grid_points(videoPath, grid_size=10, frame_limit=1000, marker_id=55)
    
    if trajectories:
        print(f"Tracking completed. Tracked {len(trajectories)} points.")
        print(f"DataFrame shape: {df.shape}")