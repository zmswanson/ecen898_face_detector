import numpy as np
import cv2
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt

class FaceDetect:
    def __init__(
            self, num_bins=32, thresh_low=100, thresh_high=200, peak_thresh=None,
            rotation_angle=0, scale=1.0, reference_noise=0.0, accumulator_noise=0.0,
            reference_point: tuple[int, int]=None, ref_images: List[np.ndarray]=None,
            query_image: np.ndarray=None, query_ground_truth: tuple[int, int]=None,
            output_prefix="fd_result", output_dir="images/results", disable_pbar=False):
        self.num_bins = num_bins
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high
        self.peak_thresh = peak_thresh
        self.reference_point = None
        self.rotation_degrees = rotation_angle
        self.rotation_radians = np.deg2rad(rotation_angle)
        self.scale = scale
        self.reference_noise = reference_noise
        self.accumulator_noise = accumulator_noise
        
        self.ref_images = []
        if ref_images:
            for img in ref_images:
                tmp = np.copy(img).astype(np.float32) + reference_noise * np.random.randn(*img.shape)
                # normalize the image to have pixel values between 0 and 255
                # and convert the image to uint8 for the Canny edge detection
                tmp = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                self.ref_images.append(tmp)

        # if ref_images:
        #     self.ref_images = [
        #         np.copy(img).astype(np.float32) + accumulator_noise * np.random.randn(*img.shape)
        #         for img in ref_images
        #     ]

        if reference_point:
            self.reference_point = reference_point
        else:
            if self.ref_images:
                self.reference_point = (
                    self.ref_images[0].shape[0] // 2,
                    self.ref_images[0].shape[1] // 2
                )
            else:
                self.reference_point = (0, 0)

        self.r_table = {}
        tmp = np.copy(query_image).astype(np.float32) + \
            accumulator_noise * np.random.randn(*query_image.shape)
        self.query_image = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.ground_truth = query_ground_truth
        self.accumulator = None
        self.accumulator_peaks = None
        self.output_prefix = output_prefix
        self.output_dir = output_dir

        self.disable_pbar = disable_pbar
        
        # aliases to match the function names requested in the assignment
        self.buildRtable = self.construct_r_table
        self.genAccumulator = self.generate_accumulator
        self.getPeaks = self.find_peaks
        self.displayResult = self.display_results

    def calculate_gradient_direction(self, img, is_quantized=True, num_bins=None):
        """
        Calculate the gradient direction of the image by applying Sobel operators in x and y
        directions and then calculating the arctangent of the two results. If quantized, then the
        gradient directions are quantized into a number of bins.

        Parameters:
        - img: The image to calculate the gradient direction of
        - is_quantized: Whether to quantize the gradient directions into n bins
        - num_bins: (Optional) The number of bins to quantize the gradient directions into

        Returns:
        - gradient_direction: The gradient direction of the image (quantized if specified)
        """
        if is_quantized:
            if num_bins is None:
                if self.num_bins is None:
                    raise ValueError("Number of bins is not defined")
            else:
                self.num_bins = num_bins

        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        gradient_direction = np.arctan2(sobel_y, sobel_x)
        
        if is_quantized:

            scaled_gd = gradient_direction % (2 * np.pi)
            bin_width = 2 * np.pi / self.num_bins
            quantized_gd = np.floor(gradient_direction / bin_width)
            gradient_direction = quantized_gd.copy()

        return gradient_direction
    
    def construct_r_table(self, ref_images=None, reference_point=None, num_bins=None):
        """
        Construct the R-table for the given reference images. The R-table is a dictionary where the
        keys are the quantized gradient directions and the values are the displacements of the edges
        from the reference point.

        Parameters:
        - ref_images: The reference images to construct the R-table from
        - reference_point: The reference point to calculate the displacements from
        - num_bins: The number of bins to quantize the gradient directions into

        Returns:
        - r_table: The R-table constructed from the reference images
        """
        if ref_images:
            self.ref_images = [np.copy(img) for img in ref_images]
        elif not self.ref_images or len(self.ref_images) == 0:
            raise ValueError("No reference images provided")

        if reference_point:
            self.reference_point = reference_point
        elif not self.reference_point:
            self.reference_point = (
                self.ref_images[0].shape[0] // 2,
                self.ref_images[0].shape[1] // 2
            )

        if num_bins:
            self.num_bins = num_bins
        elif not self.num_bins:
            raise ValueError("Number of bins is not defined")
        
        # clear the R-table before constructing a new one
        self.r_table = {}

        with tqdm(total=sum([img.shape[0] * img.shape[1] for img in self.ref_images]), disable=self.disable_pbar) as pbar:
            for img in self.ref_images:
                gradient_direction = self.calculate_gradient_direction(img, is_quantized=True)

                # openCV Canny edge detection gives a binary image with edges as white pixels (255)
                edges = cv2.Canny(img, self.thresh_low, self.thresh_high)

                for i in range(edges.shape[0]):
                    for j in range(edges.shape[1]):
                        if edges[i, j] == 255:
                            key = (gradient_direction[i, j],)

                            displacement = (
                                self.reference_point[0] - i, 
                                self.reference_point[1] - j
                            )

                            if key in self.r_table:
                                self.r_table[key].append(displacement)
                            else:
                                self.r_table[key] = [displacement]

                        pbar.update(1)

        return self.r_table

    def generate_accumulator(self, query_image=None, rotation_angle=None, scale=None):
        """
        Generate the accumulator array from the test image using the R-table. The accumulator array
        is a 2D array where each cell represents the number of votes for a particular displacement
        of the reference point.

        Parameters:
        - query_image: The image to generate the accumulator array from

        Returns:
        - accumulator: The accumulator array generated from the test image
        """
        if query_image is not None:
            self.query_image = np.copy(query_image)
        elif self.query_image is None:
            raise ValueError("No test image provided")
        
        if self.r_table is None or len(self.r_table) == 0:
            self.construct_r_table()

        if rotation_angle is not None:
            self.rotation_degrees = rotation_angle
            self.rotation_radians = np.deg2rad(rotation_angle)
        if scale is not None:
            self.scale = scale

        if self.rotation_radians is None:
            self.rotation_radians = 0
            self.rotation_degrees = 0
        if self.scale is None:
            self.scale = 1.0
        
        gradient_direction = self.calculate_gradient_direction(self.query_image)
        edges = cv2.Canny(self.query_image, self.thresh_low, self.thresh_high)

        self.accumulator = np.zeros(self.query_image.shape)

        with tqdm(total=edges.shape[0] * edges.shape[1], disable=self.disable_pbar) as pbar:
            for i in range(edges.shape[0]):
                for j in range(edges.shape[1]):
                    if edges[i, j] == 255:
                        key = (gradient_direction[i, j],)
                        if key in self.r_table:
                            for r_vect in self.r_table[key]:
                                r_row = r_vect[0]
                                r_col = r_vect[1]

                                r_row_rot = r_col * np.sin(self.rotation_radians) + \
                                    r_row * np.cos(self.rotation_radians)
                                r_col_rot = r_col * np.cos(self.rotation_radians) - \
                                    r_row * np.sin(self.rotation_radians)
                                row = int(i + (r_row_rot * self.scale))
                                col = int(j + (r_col_rot * self.scale))

                                if 0 <= row < self.accumulator.shape[0]:
                                    if 0 <= col < self.accumulator.shape[1]:
                                        self.accumulator[row, col] += 1

                    pbar.update(1)

        return self.accumulator
    
    def find_peaks(self, accumulator=None, threshold=None):
        """
        Find the peaks in the accumulator array that are above a certain threshold. The peaks are
        the points in the accumulator array that have the maximum number of votes.

        Parameters:
        - accumulator: The accumulator array to find the peaks in
        - threshold: The threshold to consider a point as a peak

        Returns:
        - peaks: The peaks in the accumulator array
        """
        if accumulator is not None:
            self.accumulator = np.copy(accumulator)
        elif self.accumulator is None:
            raise ValueError("No accumulator array provided")
        
        if threshold is None:
            if self.peak_thresh is None:
                threshold = 0.99 * np.max(self.accumulator)
            else:
                threshold = self.peak_thresh
        else:
            self.peak_thresh = threshold

        self.accumulator_peaks = []

        for i in range(self.accumulator.shape[0]):
            for j in range(self.accumulator.shape[1]):
                if self.accumulator[i, j] > threshold:
                    self.accumulator_peaks.append((self.accumulator[i, j], i, j))

        # sort the peaks in descending order of votes
        self.accumulator_peaks = sorted(self.accumulator_peaks, key=lambda x: x[0], reverse=True)

        return self.accumulator_peaks
    
    def visualize_r_table(self, write_to_file=False):
        """
        Visualize the R-table constructed from the reference images. The R-table is a dictionary
        where the keys are the quantized gradient directions and the values are the displacements
        of the edges from the reference point.
        """
        if self.r_table is None or len(self.r_table) == 0:
            raise ValueError("No R-table provided")

        rt_img = np.zeros((self.ref_images[0].shape[0], self.ref_images[0].shape[1]))

        for key, value in self.r_table.items():
            for displacement in value:
                row = self.reference_point[0] - displacement[0]
                col = self.reference_point[1] - displacement[1]

                rt_img[row, col] = 255

        if write_to_file:
            filename = self.output_prefix + \
                f"_Rtable_n{self.num_bins}_l{self.thresh_low}_h{self.thresh_high}" + \
                f"_r{self.rotation_degrees}_s{self.scale}" + \
                f"_rN{self.reference_noise}_aN{self.accumulator_noise}.png"
            wr_path = self.output_dir + filename
            cv2.imwrite(wr_path, rt_img)
        else:
            plt.imshow(rt_img, cmap='gray')
            plt.title("R-Table")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    def visualize_accumulator(self, write_to_file=False):
        """
        Visualize the accumulator array generated from the test image. The accumulator array is a
        2D array where each cell represents the number of votes for a particular displacement of
        the reference point.
        """
        if self.accumulator is None:
            raise ValueError("No accumulator array provided")
        
        if write_to_file:
            # add a hot colormap to the accumulator array using OpenCV
            accumulator_img = cv2.applyColorMap(
                cv2.convertScaleAbs(self.accumulator, alpha=255/self.accumulator.max()), cv2.COLORMAP_HOT
            )
            filename = self.output_prefix + \
                f"_accumulator_n{self.num_bins}_l{self.thresh_low}_h{self.thresh_high}" + \
                f"_r{self.rotation_degrees}_s{self.scale}" + \
                f"_rN{self.reference_noise}_aN{self.accumulator_noise}.png"
            wr_path = self.output_dir + filename
            cv2.imwrite(wr_path, accumulator_img)
        else:
            plt.imshow(self.accumulator, cmap='hot')
            plt.title("Accumulator Array")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    def visualize_peaks(self, write_to_file=False):
        """
        Visualize the peaks in the accumulator array. The peaks are the points in the accumulator
        array that have the maximum number of votes.
        """        
        peaks_img = np.copy(self.query_image)
        if len(peaks_img.shape) == 2:
            peaks_img = cv2.cvtColor(peaks_img, cv2.COLOR_GRAY2RGB)

        
        if self.accumulator_peaks is None or len(self.accumulator_peaks) == 0:
            # display a message if no peaks are detected
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(peaks_img, f"No Peaks Detected @ {self.peak_thresh} Threshold",
                        (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            for peak in self.accumulator_peaks:
                cv2.circle(peaks_img, (peak[2], peak[1]), 5, (0, 0, 255), -1)

        if write_to_file:
            filename = self.output_prefix + \
                f"_peaks_n{self.num_bins}_l{self.thresh_low}_h{self.thresh_high}" + \
                f"_r{self.rotation_degrees}_s{self.scale}" + \
                f"_rN{self.reference_noise}_aN{self.accumulator_noise}.png"
            wr_path = self.output_dir + filename
            cv2.imwrite(wr_path, peaks_img)
        else:
            plt.imshow(peaks_img)
            plt.title("Accumulator Peaks")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    def visualize_votes_histogram(self, write_to_file=False):
        """
        Visualize the histogram of votes in the accumulator array. The histogram shows the number of
        points with a binned number of votes.
        """
        if self.accumulator is None:
            raise ValueError("No accumulator array provided")
        
        votes = self.accumulator.flatten()
        plt.hist(votes, bins=100, color='b', alpha=0.7)
        plt.title("Votes Histogram")
        plt.xlabel("Number of Votes")
        plt.ylabel("Number of Points")
        plt.yscale('log')
        plt.axhline(y=1, color='r', linestyle='-')
        plt.tight_layout()
        if write_to_file:
            filename = self.output_prefix + \
                f"_votes_hist_n{self.num_bins}_l{self.thresh_low}_h{self.thresh_high}" + \
                f"_r{self.rotation_degrees}_s{self.scale}" + \
                f"_rN{self.reference_noise}_aN{self.accumulator_noise}.png"
            wr_path = self.output_dir + filename
            plt.savefig(wr_path)
        else:
            plt.show()


    def display_results(self, write_to_file=False):
        """
        Display the results of the face detection algorithm. The results include the accumulator
        array, the peaks in the accumulator array, and the ground truth if available.
        """
        if self.accumulator is None:
            raise ValueError("No accumulator array provided")
        
        boxed_img = np.copy(self.query_image)
        if len(boxed_img.shape) == 2:
            boxed_img = cv2.cvtColor(boxed_img, cv2.COLOR_GRAY2RGB)

        if self.accumulator_peaks:
            # draw a green bounding box (360,280 and scaled by 'scale' ) around the detected face 
            # of the first peak
            peak = self.accumulator_peaks[0]
            row_top = int(peak[1] - 180 * self.scale)
            row_bottom = int(peak[1] + 180 * self.scale)
            col_left = int(peak[2] - 140 * self.scale)
            col_right  = int(peak[2] + 140 * self.scale)
            
            cv2.rectangle(boxed_img, (col_left, row_top), (col_right, row_bottom), (0, 255, 0), 2)

            # draw a blue dot at the center of the detected face
            cv2.circle(boxed_img, (peak[2], peak[1]), 10, (255, 0, 0), -1)
        else:
            # display a message if no peaks are detected
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(boxed_img, f"No Peaks Detected @ {self.peak_thresh} Threshold",
                        (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if write_to_file:
            filename = self.output_prefix + \
                f"_face_result_n{self.num_bins}_l{self.thresh_low}_h{self.thresh_high}" + \
                f"_r{self.rotation_degrees}_s{self.scale}" + \
                f"_rN{self.reference_noise}_aN{self.accumulator_noise}.png"
            wr_path = self.output_dir + filename
            cv2.imwrite(wr_path, boxed_img)

        else:
            plt.imshow(boxed_img)
            plt.title("Query Image")
            plt.axis('off')

        plt.show()

    def calculate_rmse(self, ground_truth: tuple[int, int]=None):
        """
        Calculate the Root Mean Squared Error (RMSE) between the ground truth and the detected
        face location.

        Parameters:
        - ground_truth: The ground truth location of the face

        Returns:
        - rmse: The Root Mean Squared Error between the ground truth and the detected face location
        """
        if ground_truth is not None:
            self.ground_truth = ground_truth
        elif self.ground_truth is None:
            raise ValueError("No ground truth provided")

        if self.accumulator_peaks is None or len(self.accumulator_peaks) == 0:
            raise ValueError("No peaks detected in the accumulator array")

        # calculate the RMSE between the ground truth and the detected face location
        rmse = np.sqrt(
            (self.ground_truth[0] - self.accumulator_peaks[0][1]) ** 2 +
            (self.ground_truth[1] - self.accumulator_peaks[0][2]) ** 2
        )

        return rmse


if __name__ == "__main__":
    # load the reference images
    ref_images = [
        cv2.imread('images/reference/ref_img001.png', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/reference/ref_img002.png', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/reference/ref_img003.png', cv2.IMREAD_GRAYSCALE)
    ]

    # load the query image
    query_image = cv2.imread('images/test/test_img001.png', cv2.IMREAD_GRAYSCALE)

    # initialize the face detection object
    face_detect = FaceDetect(
        num_bins=32, thresh_low=100, thresh_high=200,
        ref_images=ref_images, query_image=query_image,
        query_ground_truth=(50, 50), scale=0.7, rotation_angle=-0.5
    )

    face_detect.buildRtable()
    face_detect.genAccumulator()
    face_detect.getPeaks()
    face_detect.displayResult()

    # construct the R-table
    # face_detect.construct_r_table()

    # face_detect.visualize_r_table(True)

    # # generate the accumulator array
    # face_detect.generate_accumulator()

    # face_detect.visualize_accumulator(True)

    # # find the peaks in the accumulator array
    # face_detect.find_peaks()

    # # display the results
    # face_detect.display_results(True)

    # print(f"RMSE: {face_detect.calculate_rmse((185, 290))}")