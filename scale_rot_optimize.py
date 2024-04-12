from face_detect import FaceDetect
import cv2
from tqdm import tqdm
import json

ref_images = [
    cv2.imread('images/reference/ref_img001.png', cv2.IMREAD_GRAYSCALE),
    cv2.imread('images/reference/ref_img002.png', cv2.IMREAD_GRAYSCALE),
    cv2.imread('images/reference/ref_img003.png', cv2.IMREAD_GRAYSCALE)
]

test_images = [
    (cv2.imread('images/test/test_img001.png', cv2.IMREAD_GRAYSCALE), (185, 290)),
    (cv2.imread('images/test/test_img002.png', cv2.IMREAD_GRAYSCALE), (120, 310)),
    (cv2.imread('images/test/test_img003.png', cv2.IMREAD_GRAYSCALE), (180, 255))
]

scale_factors = [i / 100 + 0.5 for i in range(0, 55, 5)] # 0.5 to 1.0 in 0.05 increments (11 values)
rotation_angles = [i / 10 - 5 for i in range(0, 105, 5)] # -5 to 5 in 0.5 increments (21 values)

test_results = {"t1": {}, "t2": {}, "t3": {}}

key_count = 1

with tqdm(total=len(test_images) * len(scale_factors) * len(rotation_angles)) as pbar:
    for test_img, gt in test_images:
        for scale_factor in scale_factors:
            for rotation_angle in rotation_angles:
                fd = FaceDetect(
                    ref_images=ref_images, scale=scale_factor, rotation_angle=rotation_angle,
                    num_bins=32, thresh_low=100, thresh_high=200, disable_pbar=True
                )

                fd.construct_r_table()

                fd.generate_accumulator(test_img)
                fd.find_peaks()
                rmse = fd.calculate_rmse(gt)

                test_results[f't{key_count}'][(scale_factor, rotation_angle)] = rmse

                pbar.update(1)

        key_count += 1

# print(test_results)

sorted_test_results = {}

# sort the results within each test image by RMSE
for test_image, results in test_results.items():
    sorted_test_results[test_image] = dict(sorted(results.items(), key=lambda item: item[1]))

# print the best results for each test image
for test_image, results in sorted_test_results.items():
    print(f'Test image {test_image}:')
    print(f'Best scale factor and rotation angle: {list(results.keys())[0]}')
    print(f'RMSE: {list(results.values())[0]}')
    print()


with open('scale_rot_results.txt', 'w') as f:
    for key, value in sorted_test_results.items():
        f.write(f'{key}: \n')

        for k, v in value.items():
            f.write(f'    {k}: {v}\n')
