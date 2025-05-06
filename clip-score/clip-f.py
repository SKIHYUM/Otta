#!/usr/bin/env python3
"""
extract_frames.py

A simple script to extract every frame from an MP4 video and save them
as PNG images in an output directory.
"""

import cv2
import os
import argparse

def extract_frames(video_path: str, output_dir: str):
    """
    Extracts frames from the given video file and writes them to output_dir.

    Args:
        video_path (str): Path to the input video file (e.g., .mp4).
        output_dir (str): Directory to save extracted frames.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames

        # Build filename and write image
        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract all frames from a video file into an output directory."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file (e.g., input.mp4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directory to save extracted frames (default: ./output)"
    )
    args = parser.parse_args()
    extract_frames(args.video, args.output)
