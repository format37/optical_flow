import cv2 as cv
import numpy as np

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

def track_grid_points(videoPath, grid_size=10):
    """
    Track a grid of points using Lucas-Kanade optical flow
    
    Parameters:
    videoPath: str, path to the video file
    grid_size: int, number of points in each dimension (creates grid_size x grid_size points)
    """
    # Read video
    cap = cv.VideoCapture(videoPath)
    if not cap.isOpened():
        print(f"Error: Could not open video at {videoPath}")
        return None
        
    # Read first frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read first frame")
        cap.release()
        return None
    
    # Create grid points
    p0 = create_grid_points(frame, grid_size)
    total_points = len(p0)
    print(f"Tracking {total_points} points in a {grid_size}x{grid_size} grid")
    
    # Convert first frame to grayscale
    old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Create a mask for drawing trajectories
    mask = np.zeros_like(frame)
    
    # Create random colors for each point
    colors = np.random.randint(0, 255, (total_points, 3)).tolist()
    
    # Set Lucas-Kanade parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Store trajectories for all points
    trajectories = [[] for _ in range(total_points)]
    for i, point in enumerate(p0):
        trajectories[i].append(tuple(map(int, point[0])))
    
    # Get video properties for output
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    
    # Create VideoWriter object
    output_path = 'output.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Calculate optical flow for all points
            p1, status, err = cv.calcOpticalFlowPyrLK(
                old_gray, 
                frame_gray, 
                p0, 
                None, 
                **lk_params
            )
            
            if p1 is not None:
                # Update each point
                for i, (new, st) in enumerate(zip(p1, status)):
                    if st[0] == 1:  # If point was found
                        new_point = tuple(map(int, new[0]))
                        old_point = tuple(map(int, p0[i][0]))
                        
                        # Draw trajectory
                        mask = cv.line(mask, new_point, old_point, colors[i], 2)
                        frame = cv.circle(frame, new_point, 3, colors[i], -1)
                        
                        # Store trajectory
                        trajectories[i].append(new_point)
                
                # Update points
                p0 = p1[status == 1].reshape(-1, 1, 2)
                colors = [c for c, s in zip(colors, status) if s[0] == 1]
            
            # Combine frame and trajectories
            img = cv.add(frame, mask)
            
            # Write frame to output video
            out.write(img)
            
            # Display frame
            cv.imshow('Grid Point Tracking', img)
            
            # Update previous frame
            old_gray = frame_gray.copy()
            
            key = cv.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            
    except Exception as e:
        print(f"Error during tracking: {str(e)}")
    
    finally:
        cap.release()
        out.release()  # Release the video writer
        cv.destroyAllWindows()
    
    print(f"Video saved to {output_path}")
    return trajectories

if __name__ == '__main__':
    videoPath = 'assets/Convection Currents in Infrared.mp4'
    # Track points in a 10x10 grid
    trajectories = track_grid_points(videoPath, grid_size=10)
    
    if trajectories:
        print(f"Tracking completed. Tracked {len(trajectories)} points.")