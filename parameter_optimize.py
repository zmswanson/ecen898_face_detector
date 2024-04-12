from face_detect import FaceDetect
import cv2
from tqdm import tqdm

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

number_of_bins = [4, 8, 16, 32, 64]
thresholds = [(20, 100), (20, 150), (20, 200), (20, 230),
              (50, 100), (50, 150), (50, 200), (50, 230),
              (100, 150), (100, 200), (100, 230),
              (150, 200), (150, 230)]

min_avg_rmse = float('inf')
best_num_bins = None
best_thresh_low = None
best_thresh_high = None
results = {}

with tqdm(total=len(number_of_bins) * len(thresholds)) as pbar:
    for num_bins in number_of_bins:
        for thresh_low, thresh_high in thresholds:
            fd = FaceDetect(
                ref_images=ref_images, num_bins=num_bins, thresh_low=thresh_low, 
                thresh_high=thresh_high, disable_pbar=True
            )
            rmse_list = []
            avg_rmse = 0

            fd.construct_r_table()

            for test_img, gt in test_images:
                fd.generate_accumulator(test_img)
                fd.find_peaks()
                rmse = fd.calculate_rmse(gt)

                rmse_list.append(rmse)
                avg_rmse += rmse

            avg_rmse /= len(test_images)

            if avg_rmse < min_avg_rmse:
                min_avg_rmse = avg_rmse
                best_num_bins = num_bins
                best_thresh_low = thresh_low
                best_thresh_high = thresh_high

            results[(num_bins, thresh_low, thresh_high)] = (rmse_list, avg_rmse)

            pbar.set_postfix({'min_avg_rmse': min_avg_rmse})
            pbar.update(1)

print(f'Best number of bins: {best_num_bins}')
print(f'Best threshold low: {best_thresh_low}')
print(f'Best threshold high: {best_thresh_high}')
print(f'Minimum average RMSE: {min_avg_rmse}')

with open('unsorted_results.txt', 'w') as f:
    for key, value in results.items():
        f.write(f'{key}: {value}\n')

# sort results by average RMSE and write to file
sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1][1])}
with open('sorted_results.txt', 'w') as f:
    for key, value in sorted_results.items():
        f.write(f'{key}: {value}\n')