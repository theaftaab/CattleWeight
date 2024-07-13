import cv2
import numpy as np
from ultralytics import YOLO
import os
import joblib
import pandas as pd
class CattleInference:
    """
    A class for performing cattle weight estimation inferences using YOLO models.

    Attributes:
        side_keypoint_model (YOLO): Loaded YOLO model for side keypoint detection.
        rear_keypoint_model (YOLO): Loaded YOLO model for rear keypoint detection.
        side_segmentation_model (YOLO): Loaded YOLO model for side segmentation.
        rear_segmentation_model (YOLO): Loaded YOLO model for rear segmentation.
        output_dir (str, optional): Directory to save inference results (default: "inference_results").

    Methods:
        __init__(self, side_keypoint_path, rear_keypoint_path, side_segmentation_path, rear_segmentation_path, output_dir="inference_results"):
            Initializes the class with model paths and optional output directory.
        infer_images(self, side_image_path, rear_image_path):
            Performs inference on side and rear cattle images and saves results.
        infer_image(self, image_path, model, filename="result.jpg"):
            Performs inference on a single image using the specified model and saves the result.
    """

    def __init__(self, cwd, output_dir="inference_results"):
        """
        Initializes the class with model paths and optional output directory.
        """
        self.side_keypoint_model = YOLO(cwd + '/side_keypoint_model.pt')
        self.rear_keypoint_model = YOLO(cwd + '/rear_keypoint_model.pt')
        self.side_segmentation_model = YOLO(cwd + '/side_segmentation_model.pt')
        self.rear_segmentation_model = YOLO(cwd + '/rear_segmentation_model.pt')
        self.output_dir = output_dir
        self.model = joblib.load(cwd + '/linear.pkl')
        # self.side_keys = None
        # self.rear_keys = None

    def infer_keypoints(self, side_image_path, rear_image_path):
        """
        Performs inference on side and rear cattle images and saves results.

        Args:
            side_image_path (str): Path to the side cattle image.
            rear_image_path (str): Path to the rear cattle image.
        """

        self.side_keys = self.infer_image(side_image_path, self.side_keypoint_model,
                                          filename=f"{self.output_dir}/side_keypoint.jpg")
        self.rear_keys = self.infer_image(rear_image_path, self.rear_keypoint_model,
                                          filename=f"{self.output_dir}/rear_keypoint.jpg")
        return self.side_keys, self.rear_keys

    def infer_image(self, image_path, model, filename="result.jpg"):
        """
        Performs inference on a single image using the specified model and saves the result.

        Args:
            image_path (str): Path to the image file.
            model (YOLO): The YOLO model to use for inference.
            filename (str, optional): Filename to save the inference result. Defaults to "result.jpg".
        """
        image = cv2.imread(image_path)
        results = model(image)
        for result in results:
            keypoints = result.keypoints  # Keypoints object for pose outputs
            # result.show()  # display to screen
            # result.save(filename="result side keypoint.jpg")
            # print(keypoints.xy)
        return keypoints.xy

    def distance(self, kpt1, kpt2):
        """
        Calculates the Euclidean distance between two keypoints.

        Args:
            kpt1 (tuple): First keypoint (x, y) coordinates.
            kpt2 (tuple): Second keypoint (x, y) coordinates.

        Returns:
            float: Euclidean distance between the keypoints.
        """

        return ((kpt1[0] - kpt2[0]) ** 2 + (kpt1[1] - kpt2[1]) ** 2) ** 0.5

    def return_args(self, kpt1, kpt2):
        self.side_length_shoulderbone = self.distance(kpt1[1], kpt1[2])
        self.side_f_girth = self.distance(kpt1[3], kpt1[4])
        self.side_r_girth = self.distance(kpt1[7], kpt1[8])
        self.rear_width = self.distance(kpt2[2], kpt2[3])
        return [self.side_length_shoulderbone, self.side_f_girth, self.side_r_girth, self.rear_width]

    def return_pixels(self, image_path, model):
        """
        Parameters:
            self: the instance of the class
            image_path: path to the image file
            model: the model used for prediction
        Returns:
            area_list: a list of areas of objects detected in the image
        """
        results = model.predict(source=image_path, imgsz=640, conf=0.9)
        area_list = []
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # Retrieve masks as numpy arrays
            image_area = masks.shape[1] * masks.shape[2]  # Calculate total number of pixels in the image
            for i, mask in enumerate(masks):
                binary_mask = (mask > 0).astype(np.uint8) * 255  # Convert mask to binary
                color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)  # Convert binary mask to color
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the binary mask
                contour = contours[0]  # Retrieve the first contour
                area = cv2.contourArea(contour)  # Calculate the area of the object
                area_list.append(area)
        return area_list

    def predict(self, side_img, rear_img):
        # cattle_inference = CattleInference()

        kpt = self.infer_keypoints(side_img, rear_img)
        side_kpt = kpt[0][0].tolist()
        rear_kpt = kpt[1][0].tolist()
        args = self.return_args(side_kpt, rear_kpt)
        pixels = self.return_pixels(side_img, self.side_segmentation_model)
        args = args + pixels

        features = ["side_length_shoulderbone", "side_f_girth", "side_r_girth", "rear_width", "cow_pixels",
                    "sticker_pixels"]
        target = "weight"
        # model = joblib.load('linear.pkl')
        new_data = {}
        for index, i in enumerate(features):
            new_data[i] = [args[index]]
        new_df = pd.DataFrame(new_data)

        predicted_weight = self.model.predict(new_df)[0] / 2

        return predicted_weight
