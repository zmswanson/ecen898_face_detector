from face_detect import FaceDetect
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

noise_levels = [1, 2, 5, 10, 20] #, 50, 100]

ref_images = [
    cv2.imread('images/reference/ref_img001.png', cv2.IMREAD_GRAYSCALE),
    cv2.imread('images/reference/ref_img002.png', cv2.IMREAD_GRAYSCALE),
    cv2.imread('images/reference/ref_img003.png', cv2.IMREAD_GRAYSCALE)
]

test_images = [
    (cv2.imread('images/test/test_img001.png', cv2.IMREAD_GRAYSCALE), (185, 290), 0.7, -0.5),
    (cv2.imread('images/test/test_img002.png', cv2.IMREAD_GRAYSCALE), (120, 310), 0.65, -4.0),
    (cv2.imread('images/test/test_img003.png', cv2.IMREAD_GRAYSCALE), (180, 255), 0.7, -1.5)
]

results = {
    "noisy_ref": {},
    "noisy_test": {},
    "noisy_both": {}
}

with tqdm(total=(len(noise_levels) * len(test_images) * 3)) as pbar:
    for noise_level in noise_levels:
        for noise_type in ['noisy_ref', 'noisy_test', 'noisy_both']:
            errors = []
            for test_img, gt, scale, rotation in test_images:
                fd = None
                if noise_type == 'noisy_ref':
                    fd = FaceDetect(
                        ref_images=ref_images, query_image=test_img, scale=scale, rotation_angle=rotation,
                        num_bins=32, thresh_low=100, thresh_high=200, disable_pbar=True,
                        reference_noise=noise_level
                    )
                elif noise_type == 'noisy_test':
                    fd = FaceDetect(
                        ref_images=ref_images, query_image=test_img, scale=scale, rotation_angle=rotation,
                        num_bins=32, thresh_low=100, thresh_high=200, disable_pbar=True,
                        accumulator_noise=noise_level
                    )
                elif noise_type == 'noisy_both':
                    fd = FaceDetect(
                        ref_images=ref_images, query_image=test_img, scale=scale, rotation_angle=rotation,
                        num_bins=32, thresh_low=100, thresh_high=200, disable_pbar=True,
                        reference_noise=noise_level, accumulator_noise=noise_level
                    )

                fd.construct_r_table()

                fd.generate_accumulator()
                fd.find_peaks()
                rmse = fd.calculate_rmse(gt)
                errors.append(rmse)
                pbar.update(1)

            errors.append(np.mean(errors))
            results[noise_type][noise_level] = errors

print(results)

with open('noise_effect_results.txt', 'w') as f:
    for key, value in results.items():
        f.write(f'{key}: \n')

        for k, v in value.items():
            f.write(f'    {k}: {v}\n')

# plot the results on a single graph with log scale for the x-axis
plt.figure(figsize=(12, 8))
plt.title("Effect of Noise Face Detection")
plt.xlabel("Noise Level")
plt.ylabel("Euclidean Distance to Ground Truth")

for noise_type, data in results.items():
    x = list(data.keys())
    y = [d[-1] for d in data.values()]
    plt.plot(x, y, label=noise_type)

plt.xscale('log')
plt.legend()
plt.show()
