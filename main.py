from face_detect import FaceDetect
import os
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Face detection using generalized Hough transform. Takes a specified ' \
            'directory of reference images and builds an R-table from all references. The ' \
            'R-table is then used to detect faces in test image specified.')
    parser.add_argument('-d', '--directory', type=str, default='images/reference/',
                        help='Directory containing reference images (default: images/reference/)')
    parser.add_argument('-t', '--test', type=str, default='images/test/test_img001.png',
                        help='Path to test image to detect faces in (default: ' \
                            'images/test/test_img001.png)')
    parser.add_argument('-n', '--num_bins', type=int, default=32,
                        help='Number of bins to quantize gradient direction (default: 32)')
    parser.add_argument('-l', '--thresh_low', type=int, default=100,
                        help='Low threshold for Canny edge detection (default: 100)')
    parser.add_argument('-u', '--thresh_high', type=int, default=200,
                        help='High threshold for Canny edge detection (default: 200)')
    parser.add_argument('-p', '--peak_thresh', type=int, default=-1,
                        help='Threshold (number of votes) for peak detection in accumulator ' \
                            '(default: -1: find maximum peak, without thresholding)')
    parser.add_argument('-s', '--scale', type=float, default=1.0,
                        help='Scale factor for applying r table to test image (default: 1.0)')
    parser.add_argument('-r', '--rotation', type=float, default=0.0,
                        help='Rotation angle in degrees for applying r table to test image ' \
                            '(default: 0.0)')
    parser.add_argument('-rn', '--ref_noise', type=float, default=0.0,
                        help='Amount (\u03C3) of Gaussian noise added to reference images ' \
                            '(default: 0.0)')
    parser.add_argument('-tn', '--test_noise', type=float, default=0.0,
                        help='Amount (\u03C3) of Gaussian noise added to test image (default: 0.0)')
    parser.add_argument('-dp', '--disable_pbar', action='store_true',
                        help='Disable progress bar')
    parser.add_argument('-v', '--visualize', type=str, default='rapd',
                        help='Visualize the following: r-table (r), accumulator (a), peaks (p), ' \
                            'detection (d), or any combination of the four. Use any combination ' \
                            'of these letters to visualize one or multiple (default: rapd)')
    parser.add_argument('-o', '--output', type=str, default='images/results/',
                        help='Path to save the visualization images (default: images/results/)')
    parser.add_argument('-w', '--pop_up', action='store_true', help='Pop up the visualization '\
                        'images in a window instead of saving them to disk. Overrides -o option.')
    
    args = parser.parse_args()

    ref_images = []
    test_image = None

    ref_dir = args.directory

    if not os.path.exists(ref_dir):
        raise FileNotFoundError(f"Reference image directory '{ref_dir}' not found")

    test_img_path = args.test

    if not os.path.exists(test_img_path):
        raise FileNotFoundError(f"Test image '{test_img_path}' not found")
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
       
    # Load reference images from directory; ignore non-image files
    for file in os.listdir(ref_dir):
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
            img = cv2.imread(os.path.join(ref_dir, file), cv2.IMREAD_GRAYSCALE)
            ref_images.append(img)

    print(f"Loaded {len(ref_images)} reference images")

    test_image = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

    output_prefix = test_img_path.split('/')[-1].split('.')[0]

    num_bins = args.num_bins
    thresh_low = args.thresh_low
    thresh_high = args.thresh_high

    peak_thresh = None

    if args.peak_thresh > 0:
        peak_thresh = abs(args.peak_thresh)

    scale = args.scale
    rotation = args.rotation
    ref_noise = args.ref_noise
    test_noise = args.test_noise
    disable_pbar = args.disable_pbar
    visualize = args.visualize.lower()
    output_dir = args.output
    pop_up = args.pop_up

    fd = FaceDetect(
        ref_images=ref_images, query_image=test_image, num_bins=num_bins, thresh_low=thresh_low,
        thresh_high=thresh_high, peak_thresh=peak_thresh, scale=scale, rotation_angle=rotation,
        reference_noise=ref_noise, accumulator_noise=test_noise, disable_pbar=disable_pbar, 
        output_prefix=output_prefix, output_dir=output_dir
    )
    
    fd.construct_r_table()
    fd.generate_accumulator()
    fd.find_peaks()
    
    if 'r' in visualize:
        fd.visualize_r_table(write_to_file=not pop_up)
    
    if 'a' in visualize:
        fd.visualize_accumulator(write_to_file=not pop_up)
    
    if 'p' in visualize:
        fd.visualize_peaks(write_to_file=not pop_up)

    if 'd' in visualize:
        fd.display_results(write_to_file=not pop_up)


if __name__ == '__main__':
    main()
    