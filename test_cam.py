import time
import numpy as np
import cv2

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

def colorize_depth(depth):
    """
    Convert a depth map to a displayable color image.
    Handles float meters or uint16 millimeters.
    """
    if depth is None:
        return None

    d = depth.copy()

    # If float (likely meters), clip a reasonable range (0.2–2.5m) for display
    if d.dtype in (np.float32, np.float64):
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d = np.clip(d, 0.2, 2.5)
        d = ((d - 0.2) / (2.5 - 0.2) * 255.0).astype(np.uint8)
    else:
        # Assume uint16 in millimeters; clip to 0.2–2.5m
        d = np.clip(d, 200, 2500)
        d = ((d - 200) / (2500 - 200) * 255.0).astype(np.uint8)

    return cv2.applyColorMap(d, cv2.COLORMAP_JET)

config = RealSenseCameraConfig(
    serial_number_or_name="338522300202",
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    use_depth=True,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect a `RealSenseCamera` with warm-up read (default).
camera = RealSenseCamera(config)
camera.connect()

# Capture a color frame via `read()` and a depth map via `read_depth()`.
try:
    color_frame = camera.read()
    #depth_map = camera.read_depth()
    print("Color frame shape:", color_frame.shape)
    #print("Depth map shape:", depth_map.shape)

    window_color = "RealSense Color"
    #window_depth = "RealSense Depth"
    cv2.namedWindow(window_color, cv2.WINDOW_NORMAL)
    #cv2.namedWindow(window_depth, cv2.WINDOW_NORMAL)

    t0 = time.perf_counter()
    last = t0
    frames = 0
    ema_fps = None  # exponential moving average for smoother on-screen FPS

    while True:
        loop_start = time.perf_counter()

        color_frame = camera.read()        # RGB
        depth_map = camera.read_depth()    # float meters or uint16

        if color_frame is None:
            # If grab failed, skip this iteration
            continue

        # Convert RGB->BGR for OpenCV display
        color_bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

        # Depth visualization
        #depth_vis = colorize_depth(depth_map)
        #if depth_vis is None:
            # If depth is disabled or failed, create a placeholder
        #    depth_vis = np.zeros_like(color_bgr)

        # FPS measurements
        now = time.perf_counter()
        dt = now - last
        last = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

        frames += 1

        # Put FPS on color image
        cv2.putText(color_bgr, f"FPS: {ema_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_color, color_bgr)
        #cv2.imshow(window_depth, depth_vis)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_time = time.perf_counter() - t0
    avg_fps = frames / total_time if total_time > 0 else 0.0
    print(f"Captured {frames} frames in {total_time:.2f}s — average FPS: {avg_fps:.2f}")
finally:
    camera.disconnect()
    cv2.destroyAllWindows()