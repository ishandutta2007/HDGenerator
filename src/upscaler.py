import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import moviepy.editor as mp
import os

def upscale_frame(session, frame_rgb):
    """
    Upscales a single video frame using the ONNX model.
    The frame is expected to be in RGB format.
    """
    # Pre-process the frame
    img = frame_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)) # HWC to CHW
    img = np.expand_dims(img, axis=0) # Add batch dimension

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img})

    # Post-process the output
    output_img_rgb = np.squeeze(result[0])
    output_img_rgb = np.transpose(output_img_rgb, (1, 2, 0)) # CHW to HWC
    output_img_rgb = np.clip(output_img_rgb * 255.0, 0, 255).astype(np.uint8)

    return output_img_rgb

def upscale_video(video_path, output_path, model_path):
    """
    Upscales a video file by processing each frame with the ONNX model
    and combines it with the original audio.
    """
    # Load the ONNX model
    print("Loading ONNX model...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print("Model loaded.")

    # Create directories for saving frames
    base_output_path = os.path.splitext(output_path)[0]
    input_frames_path = f"{base_output_path}_input_frames"
    upscaled_frames_path = f"{base_output_path}_upscaled_frames"
    os.makedirs(input_frames_path, exist_ok=True)
    os.makedirs(upscaled_frames_path, exist_ok=True)
    print(f"Input frames will be saved to: {input_frames_path}")
    print(f"Upscaled frames will be saved to: {upscaled_frames_path}")

    # Load original video to get audio
    print("Loading original video for audio extraction...")
    original_clip = mp.VideoFileClip(video_path)
    audio_clip = original_clip.audio
    print("Audio extracted.")

    # Open the input video with OpenCV for frame processing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open input video file with OpenCV")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    upscaled_frames = []
    print(f"Processing {total_frames} frames...")
    with tqdm(total=total_frames, unit='frame') as pbar:
        frame_num = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_num += 1

            # Save input frame
            cv2.imwrite(os.path.join(input_frames_path, f"frame_{frame_num:05d}.png"), frame_bgr)

            # Convert frame from BGR (OpenCV default) to RGB (MoviePy/ONNX standard)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Upscale frame
            upscaled_frame_rgb = upscale_frame(session, frame_rgb)

            # Save upscaled frame
            upscaled_frame_bgr = cv2.cvtColor(upscaled_frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(upscaled_frames_path, f"frame_{frame_num:05d}.png"), upscaled_frame_bgr)

            upscaled_frames.append(upscaled_frame_rgb)
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()

    if not upscaled_frames:
        print("No frames were upscaled. Aborting.")
        return

    print("Creating new video clip from upscaled frames...")
    # Create new video clip from the upscaled frames
    new_clip = mp.ImageSequenceClip(upscaled_frames, fps=fps)

    print("Attaching original audio to the new video clip...")
    # Set the audio from the original clip to the new clip
    final_clip = new_clip.set_audio(audio_clip)

    print(f"Writing final video to {output_path}...")
    # Write the final video file, specifying codecs to ensure compatibility
    final_clip.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True,
        threads=4, # Add some threading for speed
        logger='bar' # Show moviepy's progress bar
    )

    original_clip.close()
    final_clip.close()
    print("Video processing finished.")
