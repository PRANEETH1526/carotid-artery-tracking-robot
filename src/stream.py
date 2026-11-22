"""Ultrasound frame stream generator for live carotid artery tracking simulation.

Interface contract for Shabrish & Maruthi (Tracking & PID Control):
  - `frame_stream(path, loop=False, fps=None)` yields 256x256 float32 preprocessed frames
    matching the exact preprocessing expected by the segmentation U-Net.
"""

import os
import time
import glob
import argparse
from typing import Generator, Optional, Union

import cv2
import numpy as np

# Import preprocess pipeline from preprocess module
try:
    from preprocess import preprocess
except ImportError:
    from src.preprocess import preprocess


def frame_stream(
    path: str,
    loop: bool = False,
    fps: Optional[float] = None,
    as_uint8: bool = False,
) -> Generator[np.ndarray, None, None]:
    """Generator that simulates a live ultrasound video stream.

    Reads from a video file (.mp4, .avi, .mkv) OR a directory of sequential images.
    Applies the standardized preprocessing pipeline (grayscale -> resize 256x256 ->
    median filter -> CLAHE -> float32 [0.0, 1.0]) on every frame before yielding.

    Args:
        path: Path to a video file or folder containing sequential image frames.
        loop: If True, loop infinitely over the video/image sequence.
        fps: Optional frame rate throttle (frames per second, e.g. 30.0). If None or <= 0,
             frames are yielded as fast as possible without artificial delay.
        as_uint8: If True, yields uint8 [0, 255] frames instead of float32 [0.0, 1.0].
                  Useful for direct display with cv2.imshow or rendering.

    Yields:
        np.ndarray: 256x256 preprocessed 2D ultrasound frame (float32 [0,1] or uint8 [0,255]).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stream source path does not exist: {path}")

    is_video = os.path.isfile(path) and path.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    )
    delay = (1.0 / fps) if (fps and fps > 0) else 0.0

    while True:
        if is_video:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise IOError(f"Could not open video stream: {path}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t_start = time.perf_counter()

                # Apply standard project preprocessing
                processed = preprocess(frame, size=(256, 256))

                if as_uint8:
                    yield (processed * 255.0).round().astype(np.uint8)
                else:
                    yield processed

                if delay > 0:
                    elapsed = time.perf_counter() - t_start
                    sleep_time = max(0.0, delay - elapsed)
                    time.sleep(sleep_time)

            cap.release()

        else:
            # Assume directory of image frames
            valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
            image_files = []
            for ext in valid_exts:
                image_files.extend(glob.glob(os.path.join(path, ext)))
                image_files.extend(glob.glob(os.path.join(path, ext.upper())))

            image_files = sorted(list(set(image_files)))
            if not image_files:
                raise ValueError(f"No valid image files found in directory: {path}")

            for img_path in image_files:
                t_start = time.perf_counter()
                raw_img = cv2.imread(img_path)
                if raw_img is None:
                    continue

                processed = preprocess(raw_img, size=(256, 256))

                if as_uint8:
                    yield (processed * 255.0).round().astype(np.uint8)
                else:
                    yield processed

                if delay > 0:
                    elapsed = time.perf_counter() - t_start
                    sleep_time = max(0.0, delay - elapsed)
                    time.sleep(sleep_time)

        if not loop:
            break


def run_demo(
    source_path: str,
    fps: float = 20.0,
    loop: bool = True,
    max_frames: Optional[int] = 200,
    save_preview_path: Optional[str] = None,
) -> None:
    """Run an interactive visual demonstration of the ultrasound frame stream.

    Args:
        source_path: Path to image folder or video file.
        fps: Target frames per second.
        loop: Whether to loop the feed.
        max_frames: Stop after this many frames (None for infinite).
        save_preview_path: Optional path to save a visual snapshot of the stream.
    """
    print(f"Starting ultrasound stream demo from: {source_path}")
    print(f"Target FPS: {fps} | Loop: {loop}")

    stream = frame_stream(source_path, loop=loop, fps=fps, as_uint8=True)

    frame_idx = 0
    t0 = time.time()
    preview_saved = False

    window_name = "Autonomous Carotid Robot - Live Ultrasound Stream"
    can_display = True

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 512, 512)
    except Exception as e:
        print(f"GUI display not available ({e}). Running in headless mode.")
        can_display = False

    try:
        for frame_uint8 in stream:
            frame_idx += 1

            # Convert 1-channel to 3-channel for HUD overlay
            display_frame = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)

            # Draw HUD indicators simulating robotic ultrasound tracking system
            hud_color = (0, 255, 0)
            cv2.putText(
                display_frame,
                f"LIVE US FEED | Frame: {frame_idx:04d}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                hud_color,
                1,
            )
            cv2.putText(
                display_frame,
                "256x256 CLAHE-Denoised",
                (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 220, 255),
                1,
            )

            # Center crosshair simulating robot probe focal target
            cx, cy = 128, 128
            cv2.drawMarker(
                display_frame,
                (cx, cy),
                (0, 180, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=1,
            )

            # Save preview if requested
            if save_preview_path and not preview_saved and frame_idx == 10:
                os.makedirs(os.path.dirname(os.path.abspath(save_preview_path)), exist_ok=True)
                cv2.imwrite(save_preview_path, display_frame)
                print(f"Stream preview snapshot saved to: {save_preview_path}")
                preview_saved = True

            if can_display:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(int(1000 / fps)) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("Stream stopped by user.")
                    break
                elif key == ord(" "):  # Spacebar to pause
                    print("Stream paused. Press spacebar to resume.")
                    while True:
                        k2 = cv2.waitKey(100) & 0xFF
                        if k2 == ord(" ") or k2 == ord("q"):
                            break

            if max_frames and frame_idx >= max_frames:
                print(f"Reached demo limit of {max_frames} frames.")
                break

    except KeyboardInterrupt:
        print("Stream interrupted by keyboard.")

    finally:
        if can_display:
            cv2.destroyAllWindows()
        elapsed = time.time() - t0
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        print(f"Stream demo finished. Processed {frame_idx} frames in {elapsed:.2f}s (Avg {avg_fps:.1f} FPS).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultrasound live frame stream simulator.")
    parser.add_argument("--demo", action="store_true", help="Run interactive stream demo.")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to video file or folder of image frames. Defaults to processed test set.",
    )
    parser.add_argument(
        "--use-dataset-b",
        action="store_true",
        help="Use unannotated Dataset B (archive (1)) as stream source.",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Stream FPS throttle.")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop stream.")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum demo frames.")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Determine default source
    if args.use_dataset_b:
        # Use Dataset B demo slice inside carotid_project
        source = os.path.join(project_root, "dataset", "dataset_b_demo")
    elif args.source:
        source = args.source if os.path.isabs(args.source) else os.path.join(project_root, args.source)
    else:
        # Default to processed test set
        test_dir = os.path.join(project_root, "dataset", "processed", "test", "images")
        if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
            source = test_dir
        else:
            # Fallback to raw images
            source = os.path.join(project_root, "dataset", "raw", "images")

    preview_out = os.path.join(project_root, "reports", "stream_preview.png")

    if args.demo:
        run_demo(
            source_path=source,
            fps=args.fps,
            loop=not args.no_loop,
            max_frames=args.max_frames,
            save_preview_path=preview_out,
        )
    else:
        print("src/stream.py loaded as module.")
        print("To run the live ultrasound simulation demo, execute:")
        print("  python src/stream.py --demo")
        print("Or with Dataset B:")
        print("  python src/stream.py --demo --use-dataset-b")
