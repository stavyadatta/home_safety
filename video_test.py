import cv2

# RTSP stream URL
rtsp_url = "rtsp://admin:admin@192.168.1.11:554"

# Output video path
output_path = "/workspace/videos/vid_test_1.mp4"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream is opened correctly
if not cap.isOpened():
    print("Error opening the RTSP stream")
    exit(1)

# Get the video's width, height, and frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec for the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create a VideoWriter object to write the frames as a video file
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Read and write the frames until the stream ends
while True:
    # Read a frame from the RTSP stream
    ret, frame = cap.read()
    print(type(frame))

    # Check if the frame was read correctly
    if not ret:
        break

    # Write the frame to the output video file
    out.write(frame)

    # Display the frame (optional)
    # cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Release the resources
print("Coming here")
# cap.release()
# out.release()
# cv2.destroyAllWindows()

