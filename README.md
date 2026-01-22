# HD Video Generator 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command-line tool to upscale standard definition (SD) videos to high definition (HD) using the powerful **Real-ESRGAN** AI model.

## Features ✨

*   **AI-Powered Upscaling:** Utilizes the `RealESR_Gx4_fp16` model to intelligently increase video resolution.
*   **Simple CLI:** Easy-to-use command-line interface for quick video processing.
*   **Efficient Processing:** Uses `opencv-python` for frame-by-frame video manipulation and `onnxruntime` for fast model inference.

## Prerequisites

*   Python 3.8+
*   Git

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/HDGenerator.git
    cd HDGenerator
    ```

2.  **Create and activate a virtual environment:**
    *   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the AI Model:**
    You need to download the `RealESR_Gx4_fp16.onnx` model.
    *   **Download Link:** You can find the model on the [ONNX Model Zoo on Hugging Face](https://huggingface.co/ai-forever/Real-ESRGAN/tree/main/weights). Look for the `RealESR_Gx4_fp16.onnx` file.
    *   **Placement:** Create a `models` directory in the root of the project (if it doesn't exist) and place the downloaded `.onnx` file inside it.

    Your project structure should look like this:
    ```
    HDGenerator/
    ├── models/
    │   └── RealESR_Gx4_fp16.onnx
    ├── src/
    │   ├── main.py
    │   └── upscaler.py
    ├── videos/
    │   └── sample.mp4
    ├── .gitignore
    ├── README.md
    └── requirements.txt
    ```

## 🚀 Usage

Run the script from the root directory of the project. For example, to upscale the sample video provided in the `videos` folder, use the following command:

```bash
python src/main.py --input "videos/sample.mp4"
```

This will save the upscaled video in the `videos` directory with a `_upscaled` suffix.

### Arguments:

*   `--input` (Required): The path to the SD video file you want to upscale.
*   `--output` (Optional): The path where the upscaled HD video will be saved. If not provided, it will be saved in the same directory as the input file with an `_upscaled` suffix.
*   `--model_path` (Optional): The path to the `.onnx` model file. Defaults to `models/RealESR_Gx4_fp16.onnx`.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License 📄

This project is licensed under the MIT License.