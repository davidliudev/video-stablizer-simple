#!/usr/bin/env python3
"""
Video Stabilizer - Simple OpenCV-based video stabilization
Provides tripod-like camera stabilization effect
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional


class VideoStabilizer:
    """
    A simple video stabilizer that uses feature tracking and trajectory smoothing
    to create a tripod-like camera effect by eliminating camera movement.
    """
    
    def __init__(self, smoothing_radius: int = 100):
        """
        Initialize the video stabilizer.
        
        Args:
            smoothing_radius: Window size for trajectory smoothing (higher = more stable)
        """
        self.smoothing_radius = smoothing_radius
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
    
    def detect_features(self, gray_frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect good features to track in the frame."""
        return cv2.goodFeaturesToTrack(gray_frame, **self.feature_params)
    
    def calculate_transforms(self, video_path: str) -> Tuple[np.ndarray, Tuple[int, int, float, int]]:
        """
        Calculate transformation between consecutive frames.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Tuple of (transforms array, video info)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height}, {frame_count} frames, {fps:.2f} fps")
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transforms = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect features in previous frame
            prev_pts = self.detect_features(prev_gray)
            
            if prev_pts is not None and len(prev_pts) > 0:
                # Track features to current frame
                curr_pts, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_pts, None, **self.lk_params
                )
                
                # Filter good points
                good_new = curr_pts[status == 1]
                good_old = prev_pts[status == 1]
                
                # Need at least 10 points for reliable estimation
                if len(good_new) >= 10:
                    # Calculate transformation matrix
                    transform, _ = cv2.estimateAffinePartial2D(good_old, good_new)
                    
                    if transform is not None:
                        # Extract dx, dy, da (translation and rotation)
                        dx = transform[0, 2]
                        dy = transform[1, 2]
                        da = np.arctan2(transform[1, 0], transform[0, 0])
                    else:
                        dx = dy = da = 0
                else:
                    dx = dy = da = 0
            else:
                dx = dy = da = 0
            
            transforms.append([dx, dy, da])
            prev_gray = gray.copy()
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Analyzed {frame_idx}/{frame_count} frames")
        
        cap.release()
        return np.array(transforms), (width, height, fps, frame_count)
    
    def smooth_trajectory(self, transforms: np.ndarray) -> np.ndarray:
        """
        Smooth the camera trajectory to eliminate jitter and movement.
        
        Args:
            transforms: Array of frame-to-frame transformations
            
        Returns:
            Smoothed transformations
        """
        # Calculate cumulative trajectory
        trajectory = np.cumsum(transforms, axis=0)
        
        # Create smoothed trajectory
        smoothed_trajectory = np.copy(trajectory)
        
        for i in range(len(trajectory)):
            start = max(0, i - self.smoothing_radius)
            end = min(len(trajectory), i + self.smoothing_radius + 1)
            smoothed_trajectory[i] = np.mean(trajectory[start:end], axis=0)
        
        # Calculate smooth transforms
        smooth_transforms = smoothed_trajectory - trajectory + transforms
        
        return smooth_transforms
    
    def apply_transforms(self, video_path: str, transforms: np.ndarray, 
                        output_path: str, video_info: Tuple[int, int, float, int]) -> None:
        """
        Apply stabilization transforms to video.
        
        Args:
            video_path: Path to input video
            transforms: Array of stabilization transforms
            output_path: Path for output video
            video_info: Video properties (width, height, fps, frame_count)
        """
        cap = cv2.VideoCapture(video_path)
        width, height, fps, frame_count = video_info
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < len(transforms):
                dx, dy, da = transforms[frame_idx]
                
                # Create transformation matrix
                transform_matrix = np.array([
                    [np.cos(da), -np.sin(da), dx],
                    [np.sin(da), np.cos(da), dy]
                ], dtype=np.float32)
                
                # Apply transformation
                stabilized_frame = cv2.warpAffine(
                    frame, transform_matrix, (width, height),
                    borderMode=cv2.BORDER_REFLECT_101
                )
            else:
                stabilized_frame = frame
            
            out.write(stabilized_frame)
            frame_idx += 1
            
            if frame_idx % 10 == 0:
                print(f"Stabilized {frame_idx}/{frame_count} frames")
        
        cap.release()
        out.release()
    
    def stabilize_video(self, input_path: str, output_path: str, 
                       progress_callback: Optional[callable] = None) -> None:
        """
        Complete video stabilization process.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output stabilized video
            progress_callback: Optional callback function for progress updates
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        print(f"Starting stabilization...")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Smoothing radius: {self.smoothing_radius}")
        
        # Step 1: Calculate frame-to-frame transforms
        print("\n1. Calculating frame transformations...")
        transforms, video_info = self.calculate_transforms(input_path)
        
        # Step 2: Smooth trajectory
        print("\n2. Smoothing camera trajectory...")
        smooth_transforms = self.smooth_trajectory(transforms)
        
        # Step 3: Apply stabilization
        print("\n3. Applying stabilization...")
        self.apply_transforms(input_path, smooth_transforms, output_path, video_info)
        
        print(f"\n✅ Stabilization complete!")
        print(f"Output saved to: {output_path}")


def create_stabilizer(smoothing_radius: int = 100) -> VideoStabilizer:
    """
    Factory function to create a VideoStabilizer instance.
    
    Args:
        smoothing_radius: Higher values = more stable (tripod-like effect)
                         Recommended: 50-200
    
    Returns:
        VideoStabilizer instance
    """
    return VideoStabilizer(smoothing_radius=smoothing_radius)