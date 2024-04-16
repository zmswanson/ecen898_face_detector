# ECEN-898: Intro to Computer Vision (Python 3 - Face Detection and Generalized Hough Transform)

## Description
Python class to implement face detection using the generalized Hough transform. The class builds
an R-table from a directory of reference images and uses the R-table to detect faces in the
specified query (test) image. The class is wrapped in a command-line interface (CLI) that allows 
the user to specify reference and query images and a number of other parameters related to 

## Installation
1. Clone the repository or copy the zipped file to your desktop.
2. If you do not have Python installed, please install the latest version from
   https://www.python.org/downloads/.
3. (Optional) If you want to work from a virtual environment, please setup and activate your virtual
   environment using the virtual environment tool of your choice. For example,
   ```shell
   # Linux and Mac
   python3 -m pip install virtualenv
   python3 -m virtualenv venv
   source ./venv/bin/activate
   
   # Windows
   python -m pip install virtualenv
   python -m virtualenv venv
   venv\Scripts\activate
   ```
4. Install the required dependencies by running the following command:
    ```shell
    pip install -r requirements.txt
    ```

## Usage
The CLI offers several options for face detection:

```python
python main.py [-h] [-rd REF_DIR] [-t {1,2,3}] [-i INPUT] [-n NUM_BINS] 
                [-l THRESH_LOW] [-u THRESH_HIGH] [-p PEAK_THRESH]
                [-s SCALE] [-r ROTATION] [-rn REF_NOISE] [-tn TEST_NOISE]
                [-dp] [-v VISUALIZE] [-o OUTPUT] [-w]
```

## Options
-rd, --ref_dir: Directory containing reference images (default: images/reference/)

-t, --test_image: Select one of the three test images (default: 1), overrides the -i option

-i, --input: Path to non-standard test images for face detection (default: 
images/test/test_img001.png)

-n, --num_bins: Number of bins to quantize gradient direction (default: 32)

-l, --thresh_low: Low threshold for Canny edge detection (default: 100)

-u, --thresh_high: High threshold for Canny edge detection (default: 200)

-p, --peak_thresh: Threshold (number of votes) for peak detection in accumulator (default: -1:
find maximum peak, without thresholding)

-s, --scale: Scale factor for applying r table to test image (default: 1.0)

-r, --rotation: Rotation angle in degrees for applying r table to test image (default: 0.0)

-rn, --ref_noise: Amount (σ) of Gaussian noise added to reference images (default: 0.0)

-tn, --test_noise: Amount (σ) of Gaussian noise added to test image (default: 0.0)

-dp, --disable_pbar: Disable progress bar

-v, --visualize: Visualize the following: r-table (r), accumulator (a), peaks (p), votes histogram
(h), detection (d), or any combination of the four. Use any combination of these letters to
visualize one or multiple (default: rapd)

-o, --output: Path to save the visualization images (default: images/results/)

-w, --pop_up: Pop up the visualization images in a window instead of saving them to disk.
Overrides -o option.

## Example

To perform face detection using default settings:

```python
python main.py
```

To specify a custom test image and visualize the accumulator and peaks without saving to disk:

```python
python face_detection.py -t 2 -v ap -w
```

## Other Applications

Three other Python scripts were also developed for optimizing parameters, optimizing rotation and 
scale, and studying the effects of noise. These scripts are *parameter_optimize.py*, 
*scale_rot_optimize.py*, and *noise_effects.py*, respectively

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free
to use, modify, and distribute the code in this repository for any purpose, including commercial
applications, as long as you include the original copyright notice and the license terms.