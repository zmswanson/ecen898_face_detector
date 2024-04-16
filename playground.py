# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# %%
num_bins = 32
thresh_low = 100
thresh_high = 200

# %%
def quantize_gradient_direction(gradient_direction, num_bins, is_degrees=False):
    if is_degrees:
        gradient_direction = np.deg2rad(gradient_direction)
        
    gradient_direction = gradient_direction % (2 * np.pi)

    bin_width = 2 * np.pi / num_bins
    quantized_direction = np.floor(gradient_direction / bin_width)
    return quantized_direction

# %%
def compute_gradient_direction(img, is_quantized=False, num_bins=8):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    gradient_direction = np.arctan2(sobel_y, sobel_x)
    
    if is_quantized:
        gradient_direction = quantize_gradient_direction(gradient_direction, num_bins)

    return gradient_direction

# %%
def construct_r_table(ref_images, reference_point: tuple[int, int], num_bins):
    r_table = {}

    with tqdm(total=sum([img.shape[0] * img.shape[1] for img in ref_images])) as pbar:
        for img in ref_images:
            # img = cv2.GaussianBlur(img, (5, 5), 2)
            gradient_direction = compute_gradient_direction(img, is_quantized=True, num_bins=num_bins)
            edges = cv2.Canny(img, thresh_low, thresh_high)

            for i in range(edges.shape[0]):
                for j in range(edges.shape[1]):
                    if edges[i, j] == 255:
                        key = (gradient_direction[i, j],)
                        displacement = (reference_point[0] - i, reference_point[1] - j)
                        if key in r_table:
                            r_table[key].append(displacement)
                        else:
                            r_table[key] = [displacement]
                    pbar.update(1)

    return r_table

# %%
ref1 = cv2.imread('images/reference/ref_img001.png', cv2.IMREAD_GRAYSCALE)
ref2 = cv2.imread('images/reference/ref_img002.png', cv2.IMREAD_GRAYSCALE)
ref3 = cv2.imread('images/reference/ref_img003.png', cv2.IMREAD_GRAYSCALE)
ref_images = [ref1, ref2, ref3]

# ref_images = [ref_img]
refence_point = (ref1.shape[0] // 2, ref1.shape[1] // 2)
r_table = construct_r_table(ref_images, refence_point, num_bins)

# %% visualize r_table by plotting the reference image overlayed with the points in the r_table
# with a color corresponding to the gradient direction bin
def visualize_r_table(r_table, ref_img, reference_point, num_bins):
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)

    # fade the reference image so that the points are more visible
    ref_img = cv2.addWeighted(ref_img, 0.5, np.zeros_like(ref_img), 0.5, 0)
    colors = []

    for key, value in r_table.items():
        color = np.random.randint(0, 255, 3)
        colors.append(color)
        for point in value:
            point = (reference_point[0] - point[0], reference_point[1] - point[1])
            ref_img[point[0], point[1]] = color
            # point = (point[0] + reference_point[0], point[1] + reference_point[1])
            # ref_img[point[0], point[1]] = color

    # add a circle around the reference point
    reference_point = (reference_point[1], reference_point[0])
    cv2.circle(ref_img, reference_point, 10, (0, 0, 255), 2)


    # add a colorbar to show the mapping between gradient direction and color
    plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    # plt.colorbar(plt.imshow(np.arange(num_bins).reshape(1, num_bins), cmap='hsv', aspect=10))
    plt.show()

# %%
print(refence_point, ref1.shape)
visualize_r_table(r_table, ref1, refence_point, num_bins)
visualize_r_table(r_table, ref2, refence_point, num_bins)
visualize_r_table(r_table, ref3, refence_point, num_bins)


# %%
def generate_accumulator(r_table, query_img, num_bins):
    accumulator = np.zeros_like(query_img)

    # query_img = cv2.GaussianBlur(query_img, (5, 5), 2)
    gradient_direction = compute_gradient_direction(query_img, is_quantized=True, num_bins=num_bins)
    edges = cv2.Canny(query_img, thresh_low, thresh_high)

    with tqdm(total=edges.shape[0] * edges.shape[1]) as pbar:
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                if edges[i, j] == 255:
                    key = (gradient_direction[i, j],)
                    if key in r_table:
                        for point in r_table[key]:
                            r = i + point[0]
                            c = j + point[1]
                            if 0 <= r < accumulator.shape[0] and 0 <= c < accumulator.shape[1]:
                                accumulator[r, c] += 1

                pbar.update(1)

    return accumulator

# %%
query_img = cv2.imread('images/test/test_img002.png', cv2.IMREAD_GRAYSCALE)
accumulator = generate_accumulator(r_table, query_img, num_bins)

# %% visualize the accumulator as a histogram by the number of points with a binned number of votes
plt.hist(accumulator.flatten(), bins=100)
# draw a line at y=1 to show the number of points with 1 vote
plt.axhline(y=1, color='r', linestyle='-')
plt.yscale('log')
plt.show()

# %% import kmeans clustering
from sklearn.cluster import KMeans

def find_maxima(accumulator, num_maxima):
    """
    Find the maxima in the accumulator that are above a certain threshold. Then, use k-means
    clustering to find the coordinates of the maxima.
    """
    maxima = []
    max_val = np.max(accumulator)
    print(max_val)
    threshold = 0.99 * max_val
    print(threshold)

    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
            if accumulator[i, j] > threshold:
                maxima.append((accumulator[i, j], i, j))

    print(len(maxima))
    maxima = np.array([[x[1], x[2]] for x in maxima])

    plt.imshow(accumulator, cmap='hot')
    plt.scatter(maxima[:, 1], maxima[:, 0], c='blue')
    plt.show()

    kmeans = KMeans(n_clusters=num_maxima, random_state=0).fit(maxima)

    # return the coordinates of the cluster centers as int tuples
    return kmeans.cluster_centers_.astype(int)

# %%
maxima = find_maxima(accumulator, 1)
print(maxima)
# %% draw a circle around the maxima in the query image
query_copy = cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR)
print(query_img.shape)
for point in maxima:
    point = (point[1], point[0])
    cv2.circle(query_copy, point, 10, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(query_copy, cv2.COLOR_BGR2RGB))
plt.show()



# %%
def visualize_accumulator(accumulator, img=None):
    if img is not None:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = np.zeros((accumulator.shape[0], accumulator.shape[1], 3))

    max_val = np.max(accumulator)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.imshow(accumulator, cmap='hot', alpha=0.5, vmin=0, vmax=max_val)
    plt.colorbar()
    plt.show()

# %%
visualize_accumulator(accumulator, query_copy)

# %%
results = {
    'noisy_ref': {
        1: [1.0, 1.0, 4.0, 2.0], 
        2: [1.0, 1.0, 4.123105625617661, 2.04103520853922], 
        5: [4.123105625617661, 11.40175425099138, 4.123105625617661, 6.549321834075567], 
        10: [6.082762530298219, 9.219544457292887, 7.0, 7.434102329197035], 
        20: [10.770329614269007, 1.0, 3.1622776601683795, 4.9775357581457955]}, 
    'noisy_test': {
        1: [1.0, 1.0, 4.123105625617661, 2.04103520853922], 
        2: [2.0, 1.0, 4.123105625617661, 2.374368542], 
        5: [4.123105625617661, 10.04987562112089, 4.123105625617661, 6.098695624118737], 
        10: [2.23606797749979, 2.8284271247461903, 8.246211251235321, 4.4369021178271], 
        20: [2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979]}, 
    'noisy_both': {
        1: [1.0, 1.0, 3.605551275463989, 1.8685170918213299], 
        2: [4.0, 2.0, 7.0, 4.333333333333333], 
        5: [9.219544457292887, 1.0, 3.605551275463989, 4.608365244252292], 
        10: [2.0, 9.0, 2.23606797749979, 4.412022659166596], 
        20: [16.0, 4.123105625617661, 3.1622776601683795, 7.761794428595347]}}

# %%
# plot the results on a single graph with log scale for the x-axis
plt.figure(figsize=(12, 8))
plt.title("Effect of Noise Face Detection")
plt.xlabel("Noise Level")
plt.ylabel("Euclidean Distance to Ground Truth")

for noise_type, data in results.items():
    x = list(data.keys())
    y = [d[-1] for d in data.values()]
    plt.plot(x, y, label=noise_type)

# plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.show()
# %%
