import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Construct an `OpenCVCameraConfig` with your desired FPS, resolution, color mode, and rotation.
config = OpenCVCameraConfig(
    index_or_path="/dev/video0",
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect an `OpenCVCamera`, performing a warm-up read (default).
camera = OpenCVCamera(config)
camera.connect()

win_name = "OpenCV Camera (LeRobot)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Read frames asynchronously in a loop via `async_read(timeout_ms)`
try:
    while True:
        frame = camera.async_read(timeout_ms=200)

        if frame is None:
            print(f"[WARN] Timeout/no frame at iteration")
            continue
        
        color_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w = frame.shape[:2]

        # Centered square rectangle (max 250x250, but never more than half the frame)
        rect_size = min(25, h // 2, w // 2)
        x1 = w // 2 - rect_size // 2
        y1 = h // 2 - rect_size // 2
        x2 = x1 + rect_size
        y2 = y1 + rect_size

        # Draw rectangle (thickness=2)
        cv2.rectangle(color_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show
        cv2.imshow(win_name, color_bgr)

        # Exit early on ESC (27), or continue after ~1ms delay
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("ESC pressed, exiting.")
            break
        elif key == ord("s"):  # Save frame
            filename = f"real.png"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

finally:
    camera.disconnect()