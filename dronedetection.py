import argparse
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "tracking_results"

MODEL_CONFIG = BASE_DIR / "yolov3.cfg"
MODEL_WEIGHTS = BASE_DIR / "yolov3.weights"

CONFIDENCE_THRESHOLD = 0.50
NMS_THRESHOLD = 0.40
INPUT_SIZE = (416, 416)

# Standard YOLOv3 COCO class 47 = apple, NOT drone.
# The standard COCO model has no drone class.
# None means display detections from all 80 COCO classes.
# A drone-trained model is required for real drone detection.
TARGET_CLASS_IDS = None


# ============================================================
# LOAD YOLO MODEL
# ============================================================

def load_model():
    """Load the YOLOv3 network once."""

    if not MODEL_CONFIG.is_file():
        raise FileNotFoundError(
            f"Missing configuration file: {MODEL_CONFIG}"
        )

    if not MODEL_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Missing weights file: {MODEL_WEIGHTS}"
        )

    print("Loading YOLOv3 model...")

    # General-purpose loader for Darknet weights and config.
    net = cv2.dnn.readNet(
        str(MODEL_WEIGHTS),
        str(MODEL_CONFIG),
    )

    # Use CPU inference for broad macOS compatibility.
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getUnconnectedOutLayersNames()

    print("YOLOv3 model loaded successfully!")

    return net, layer_names


# ============================================================
# OBJECT DETECTION
# ============================================================

def detect_objects(frame, net, layer_names):
    """
    Run YOLOv3 inference and draw bounding boxes.

    Note: Standard COCO YOLOv3 is not trained specifically
    to detect drones.
    """

    height, width = frame.shape[:2]

    # Convert frame into a blob for YOLO.
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1 / 255.0,
        size=INPUT_SIZE,
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )

    net.setInput(blob)

    outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    # Extract detection predictions.
    for output in outputs:
        for detection in output:

            scores = detection[5:]

            class_id = int(np.argmax(scores))

            # YOLO's objectness score multiplied by
            # the probability of the predicted class.
            confidence = (
                float(detection[4]) *
                float(scores[class_id])
            )

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            if (
                TARGET_CLASS_IDS is not None
                and class_id not in TARGET_CLASS_IDS
            ):
                continue

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)

            box_width = int(detection[2] * width)
            box_height = int(detection[3] * height)

            x = int(center_x - box_width / 2)
            y = int(center_y - box_height / 2)

            boxes.append([
                x,
                y,
                box_width,
                box_height,
            ])

            confidences.append(confidence)
            class_ids.append(class_id)

    detections = []

    # Apply Non-Maximum Suppression separately per class.
    for class_id in set(class_ids):

        matching_indices = [
            i for i, value in enumerate(class_ids)
            if value == class_id
        ]

        class_boxes = [boxes[i] for i in matching_indices]
        class_confidences = [
            confidences[i] for i in matching_indices
        ]

        indices = cv2.dnn.NMSBoxes(
            class_boxes,
            class_confidences,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD,
        )

        if len(indices) == 0:
            continue

        for index in np.asarray(indices).flatten():

            original_index = matching_indices[int(index)]

            x, y, box_width, box_height = boxes[original_index]
            confidence = confidences[original_index]

            x1 = max(0, x)
            y1 = max(0, y)

            x2 = min(width, x + box_width)
            y2 = min(height, y + box_height)

            if x2 <= x1 or y2 <= y1:
                continue

            label = (
                f"COCO class {class_id}: "
                f"{confidence:.1%}"
            )

            # Draw bounding box.
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # Draw detection label.
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            detections.append({
                "class_id": class_id,
                "confidence": confidence,
                "box": [
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                ],
            })

    return frame, detections


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_videos(
    input_dir,
    output_dir,
    net,
    layer_names,
    display=False,
):
    """
    Read MP4 videos, process every frame, and save
    the annotated result as a new MP4 video.

    No individual PNG files are created.
    """

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    # Create tracking_results if it doesn't exist.
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".mp4"
    )

    if not video_files:
        print("No MP4 videos found in input directory.")
        return

    for video_path in video_files:

        print(f"\nProcessing video: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Unable to open {video_path.name}")
            continue

        # Read original video properties.
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if width <= 0 or height <= 0:
            cap.release()
            print("Invalid video dimensions. Skipping.")
            continue

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Create output MP4 path.
        output_path = (
            output_dir /
            f"{video_path.stem}_tracked.mp4"
        )

        # Initialize MP4 video writer.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not create output MP4: {output_path}"
            )

        frame_count = 0
        detection_count = 0
        frames_with_detections = 0

        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Total frames: {total_frames}")

        try:
            while True:

                success, frame = cap.read()

                if not success:
                    break

                # Run detection on current frame.
                annotated_frame, detections = detect_objects(
                    frame,
                    net,
                    layer_names,
                )

                # Write annotated frame directly to MP4.
                writer.write(annotated_frame)

                frame_count += 1

                detection_count += len(detections)

                if detections:
                    frames_with_detections += 1

                # Display progress every 100 frames.
                if frame_count % 100 == 0:
                    if total_frames > 0:
                        progress = (
                            frame_count / total_frames
                        ) * 100

                        print(
                            f"Processed {frame_count}/"
                            f"{total_frames} frames "
                            f"({progress:.1f}%)"
                        )
                    else:
                        print(
                            f"Processed {frame_count} frames"
                        )

                # Optional preview window.
                if display:

                    cv2.imshow(
                        "YOLOv3 Detection",
                        annotated_frame,
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Processing stopped by user.")
                        break

        finally:
            cap.release()
            writer.release()

        print("\n================================")
        print("VIDEO PROCESSING COMPLETE")
        print("================================")
        print(f"Video: {video_path.name}")
        print(f"Frames processed: {frame_count}")
        print(f"Total detections: {detection_count}")
        print(
            f"Frames with detections: "
            f"{frames_with_detections}"
        )
        print(f"Output saved: {output_path}")
        print("================================")

    if display:
        cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Process MP4 videos using YOLOv3 "
            "and save annotated MP4 results."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing input MP4 videos.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for processed MP4 videos.",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live detection preview.",
    )

    args = parser.parse_args()

    try:

        net, layer_names = load_model()

        process_videos(
            args.input,
            args.output,
            net,
            layer_names,
            display=args.display,
        )

    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        cv2.error,
    ) as error:

        raise SystemExit(
            f"\nError: {error}"
        ) from error


if __name__ == "__main__":
    main()
