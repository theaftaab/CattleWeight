from CattleWeight.Cattle_inference import CattleInference
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def predict(side_img, rear_img):
    cattle_inference = CattleInference(
        side_keypoint_path="CattleWeight/side_keypoint_model.pt",
        rear_keypoint_path="CattleWeight/rear_keypoint_model.pt",
        side_segmentation_path="CattleWeight/YOLO Models/side_segmentation_model/last.pt",
        rear_segmentation_path="CattleWeight/rear_segmentation_model.pt",
        output_dir="inference_results")

    kpt = cattle_inference.infer_keypoints(side_img, rear_img)
    side_kpt = kpt[0][0].tolist()
    rear_kpt = kpt[1][0].tolist()
    args = cattle_inference.return_args(side_kpt, rear_kpt)
    print(args)
    pixels = cattle_inference.return_pixels(side_img, cattle_inference.side_segmentation_model)
    print(pixels)
    args = args + pixels
    print(f'Args: {args}')

    # Load your data into a pandas DataFrame
    data = pd.read_csv("CattleWeight/train_data.csv")  # Replace with your data loading method

    # Define the features (independent variables) and target (dependent variable)
    features = ["side_length_shoulderbone", "side_f_girth", "side_r_girth", "rear_width", "cow_pixels",
                "sticker_pixels"]
    target = "weight"

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    # Create the linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model performance (optional)
    r_squared = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    new_data = {}
    for index, i in enumerate(features):
        new_data[i] = [args[index]]
    new_df = pd.DataFrame(new_data)

    predicted_weight = model.predict(new_df)[0]

    return predicted_weight


# side_img = "/Users/aftaabhussain/Work/Data Accumulator and prediction/Images/Side/127_b4-1_s_131_M.jpg"
# rear_img = "/Users/aftaabhussain/Work/Data Accumulator and prediction/Images/Rear/127_b4-1_r_131_M.jpg"
#
# weight = predict(side_img, rear_img)
