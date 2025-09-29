import cv2
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Construct an `OpenCVCameraConfig` with your desired FPS, resolution, color mode, and rotation.
config = RealSenseCameraConfig(
    serial_number_or_name="338522300202",
    fps=30,
    width=1280,
    height=720,
    color_mode=ColorMode.RGB,
    use_depth=True,
    rotation=Cv2Rotation.NO_ROTATION
)


# Instantiate and connect an `OpenCVCamera`, performing a warm-up read (default).
camera = RealSenseCamera(config)
camera.connect()

win_name = "OpenCV Camera (LeRobot)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

num = 0

# Read frames asynchronously in a loop via `async_read(timeout_ms)`
try:
    while True:
        frame = camera.read()

        if frame is None:
            print(f"[WARN] Timeout/no frame at iteration")
            continue
        
        color_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w = frame.shape[:2]
        x_mid = w // 2
        h_mid = h // 2
        cv2.line(color_bgr, (x_mid, 0), (x_mid, h), (0, 255, 0), 2)
        cv2.line(color_bgr, (0, 607), (w, 607), (0, 255, 0), 2) 

        # Show
        cv2.imshow(win_name, color_bgr)

        # Exit early on ESC (27), or continue after ~1ms delay
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("ESC pressed, exiting.")
            break
        elif key == ord("s"):  # Save frame
            filename = f"images/realsense/img_{num}.png"
            num += 1
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

finally:
    camera.disconnect()