import cv2
import matplotlib.pyplot as plt
import os

# Video file path
video_path = "/media/data1/ravram/MHI_CATH_DICOM_VIDEOS/2019/2.16.124.113611.1.118.1.1.5319172_1.3.46.670589.28.265405998058.20191018144415744231.2.2.dcm.avi"

# Check if file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit(1)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit(1)

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

# Calculate middle frame index (middle of the entire video)
middle_frame_idx = total_frames // 2  # Integer division to get the middle frame

# Set the frame position
cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)

# Read the frame
ret, frame = cap.read()

if not ret:
    print(f"Error: Could not read frame {middle_frame_idx}")
    cap.release()
    exit(1)

# Convert BGR to RGB (OpenCV uses BGR by default)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Create output directory if it doesn't exist
output_dir = "/volume/DeepCORO_CLIP/output_frames"
os.makedirs(output_dir, exist_ok=True)

# Save the frame
output_filename = os.path.join(output_dir, "middle_frame.png")
plt.figure(figsize=(12, 8))
plt.imshow(frame_rgb)
plt.title(f"Middle Frame (Frame {middle_frame_idx} of {total_frames})")
plt.axis('off')
plt.tight_layout()
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"Frame saved to: {output_filename}")

# Also display the plot
plt.show()

# Release the video capture object
cap.release()
