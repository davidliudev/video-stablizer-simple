#!/usr/bin/env python3
"""
Command-line interface for video stabilization
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from video_stabilizer import VideoStabilizer


def main():
    parser = argparse.ArgumentParser(
        description='Stabilize video to create tripod-like camera effect',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_video',
        help='Path to input video file'
    )
    
    parser.add_argument(
        'output_video',
        help='Path for output stabilized video'
    )
    
    parser.add_argument(
        '--smoothing-radius',
        type=int,
        default=100,
        help='Smoothing radius for stabilization (higher = more stable, 50-200 recommended)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Video Stabilizer 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_video):
        print(f"❌ Error: Input video not found: {args.input_video}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")
    
    # Check if output file already exists
    if os.path.exists(args.output_video):
        response = input(f"⚠️  Output file already exists: {args.output_video}\nOverwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operation cancelled")
            sys.exit(1)
    
    try:
        # Create stabilizer
        stabilizer = VideoStabilizer(smoothing_radius=args.smoothing_radius)
        
        # Perform stabilization
        stabilizer.stabilize_video(args.input_video, args.output_video)
        
        print(f"\n🎉 Success! Stabilized video saved to: {args.output_video}")
        
    except Exception as e:
        print(f"❌ Error during stabilization: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()