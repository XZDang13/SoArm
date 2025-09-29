import cv2


win_name = "OpenCV Camera (LeRobot)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

frame = cv2.imread("sim.png")

try:
    while True:
        color_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
    cv2.destroyAllWindows()