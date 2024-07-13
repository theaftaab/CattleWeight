from ultralytics import YOLO


side_keypoint_model = YOLO("/Users/aftaabhussain/Work/Cattle-weight/YOLO Models/side_keypoint_model/last.pt")  # load a custom mode
rear_keypoint_model = YOLO("/Users/aftaabhussain/Work/Cattle-weight/YOLO Models/rear_keypoint_model/last.pt")  # load a custom mode
side_segmentation_model = YOLO("/Users/aftaabhussain/Work/Cattle-weight/YOLO Models/side_segmentation_model/last.pt")  # load a custom mode
rear_segmentation_model = YOLO("/Users/aftaabhussain/Work/Cattle-weight/YOLO Models/rear_segmentation_model/last.pt")  # load a custom mode
# Predict with the model
side_image = "/Users/aftaabhussain/Work/yolo segmentation/B4/data/images/train/284_b4-2_s_140_F.jpg"

rear_image = "/Users/aftaabhussain/Work/yolo segmentation/B4/Rear/images/5_b4-1_r_77_M.jpg"
# results = side_keypoint_model(image)  # predict on an image
def infer(model, image, filename="result side keypoint.jpg"):
    results = model(image)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        print(result.masks)
        result.show()  # display to screen
        result.save(filename="result side keypoint.jpg")
        print(keypoints.xy)
infer(side_keypoint_model, side_image, filename="result side keypoint.jpg")
infer(rear_keypoint_model, rear_image, filename="result rear keypoint.jpg")
infer(side_segmentation_model, side_image, filename="result side segmentation.jpg")
infer(rear_segmentation_model, rear_image, filename="result rear segmentation.jpg")