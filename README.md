# Drone Detection with YOLOv3 and OpenCV

A simple Python project that processes MP4 videos with YOLOv3 and OpenCV, draws boxes around recognized objects, and saves the processed footage as an MP4. It does not save individual image frames.

> **Note:** The included standard YOLOv3 COCO model does **not** have a drone class. It can detect supported COCO objects, but reliable drone detection requires drone-trained weights. The `_tracked` filename is a naming convention; the current script performs frame-by-frame detection, not persistent object tracking.

## View a Sample Result

If a demo video has been uploaded to this repository, open `tracking_results/trackingtest_tracked.mp4` on GitHub and choose **Download raw file** to watch it without installing anything.

## Run It Yourself

**1. Clone the repository**

```bash
git clone https://github.com/jasmitsbawa/drone_detection.git
cd drone_detection
```

**2. Create and activate a virtual environment** (Python 3.12 recommended)

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
python -m pip install opencv-python numpy
```

**4. Download the model weights**

The repository includes `yolov3.cfg`, but the large `yolov3.weights` file is not committed. Download it into the project root:

```bash
curl -L --fail -o yolov3.weights https://data.pjreddie.com/files/yolov3.weights
```

On Windows without `curl`, download the file using that URL in a browser and save it as `yolov3.weights` beside `uavdetection.py`.

**5. Add a video and run**

Put one or more `.mp4` files in `input/`, then run:

```bash
python uavdetection.py
```

For an optional live preview:

```bash
python uavdetection.py --display
```

**6. Watch the output**

Processed videos are saved in `tracking_results/`:

```text
input/myvideo.mp4  →  tracking_results/myvideo_tracked.mp4
```

The script creates `tracking_results/` automatically if needed. Videos are exported without their original audio.

## Project Files

```text
drone_detection/
├── input/                 # Add your MP4 files here
├── tracking_results/      # Generated MP4 files (optional demo in GitHub)
├── uavdetection.py        # Main script
├── yolov3.cfg             # Model configuration
└── README.md
```

Your local `.venv/` folder and downloaded `yolov3.weights` are **not** included in Git. Create/download them using the steps above. The old `detections/` PNG folder is no longer used.

## Technologies

Python · OpenCV · NumPy · YOLOv3

**Author:** [Jasmit Bawa](https://github.com/jasmitsbawa)
