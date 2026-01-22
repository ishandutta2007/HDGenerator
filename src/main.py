import argparse
import os
from upscaler import upscale_video
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Upscale a video to HD using Real-ESRGAN.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input SD video file.")
    parser.add_argument("--output", type=str, help="Path to the output HD video file. Defaults to a file in the same directory as the input.")
    parser.add_argument("--model_path", type=str, default="models/RealESR_Gx4_fp16.onnx", help="Path to the ONNX model file.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please download the RealESR_Gx4_fp16.onnx model and place it in the 'models' directory.")
        return

    output_path = args.output
    if not output_path:
        input_dir, input_filename = os.path.split(args.input)
        input_name, input_ext = os.path.splitext(input_filename)
        output_path = os.path.join(input_dir, f"{input_name}_upscaled_{datetime.now().strftime('%Y%m%d%H%M%S')}{input_ext}")

    print(f"Upscaling video: {args.input}")
    print(f"Model: {args.model_path}")
    print(f"Output will be saved to: {output_path}")

    try:
        upscale_video(args.input, output_path, args.model_path)
        print("Video upscaling completed successfully!")
    except Exception as e:
        print(f"An error occurred during upscaling: {e}")

if __name__ == "__main__":
    main()
