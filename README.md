# Video Stabilizer Simple

A simple OpenCV-based video stabilization tool that creates a tripod-like camera effect by eliminating camera movement, panning, and zooming.

## Features

- **Tripod-like Effect**: Eliminates camera shake and movement to simulate a stationary camera
- **Feature Tracking**: Uses OpenCV's optical flow for robust motion detection
- **Trajectory Smoothing**: Applies configurable smoothing to create stable footage
- **Simple Interface**: Easy-to-use command-line tool
- **Lightweight**: Only requires OpenCV and NumPy

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python stabilize.py input_video.mp4 output_video.mp4
```

### With Custom Smoothing

```bash
python stabilize.py input_video.mp4 output_video.mp4 --smoothing-radius 150
```

### Options

- `--smoothing-radius`: Controls the level of stabilization (default: 100)
  - Higher values = more stable (tripod-like effect)
  - Recommended range: 50-200
  - Values too high may cause artifacts

## Python API

```python
from src.video_stabilizer import VideoStabilizer

# Create stabilizer with custom smoothing
stabilizer = VideoStabilizer(smoothing_radius=100)

# Stabilize video
stabilizer.stabilize_video('input.mp4', 'output.mp4')
```

## How It Works

1. **Feature Detection**: Identifies good features to track in each frame
2. **Optical Flow**: Tracks feature movement between consecutive frames
3. **Transform Calculation**: Calculates translation and rotation between frames
4. **Trajectory Smoothing**: Smooths the camera trajectory over time
5. **Stabilization**: Applies inverse transforms to eliminate camera movement

## Example Results

The stabilizer is particularly effective for:
- Handheld camera footage
- Footage with camera shake
- Creating static camera effects from moving footage
- Eliminating panning and zooming movements

## Technical Details

- **Algorithm**: Feature-based stabilization using Lucas-Kanade optical flow
- **Features**: Good Features to Track (GFTT) corner detection
- **Smoothing**: Moving average trajectory smoothing
- **Output**: Same resolution and frame rate as input

## Requirements

- Python 3.7+
- OpenCV 4.x
- NumPy

## Limitations

- Processing time depends on video length and resolution
- Very high smoothing values may cause border artifacts
- Works best with videos that have trackable features
- Cannot recover from extreme camera movements

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests!

## Examples

### Basic Usage
```bash
# Stabilize a shaky video
python stabilize.py shaky_video.mp4 stable_video.mp4

# Use higher smoothing for tripod effect
python stabilize.py handheld.mp4 tripod_like.mp4 --smoothing-radius 200
```

### Expected Output
```
Starting stabilization...
Input: input.mp4
Output: output.mp4
Smoothing radius: 100

Video info: 1280x720, 300 frames, 30.00 fps

1. Calculating frame transformations...
Analyzed 10/300 frames
...

2. Smoothing camera trajectory...

3. Applying stabilization...
Stabilized 10/300 frames
...

✅ Stabilization complete!
Output saved to: output.mp4
```