import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_funnel import CattleInference
import os
from sklearn.metrics import r2_score, mean_squared_error

# ... your existing cattle inference imports and functions
cattle_inference = CattleInference(
    side_keypoint_path="/Users/aftaabhussain/Work/Cattle-weight/YOLO Models/side_keypoint_model/last.pt",
    rear_keypoint_path="/Users/aftaabhussain/Work/Cattle-weight/YOLO Models/rear_keypoint_model/last.pt",
    side_segmentation_path="/Users/aftaabhussain/Work/Cattle-weight/YOLO Models/side_segmentation_model/last.pt",
    rear_segmentation_path="/Users/aftaabhussain/Work/Cattle-weight/YOLO Models/rear_segmentation_model/last.pt",
    output_dir="inference_results")
# Load your data into a pandas DataFrame
data = pd.read_csv("train_data.csv")  # Replace with your data loading method

# Define the features (independent variables) and target (dependent variable)
features = ["side_length_shoulderbone", "side_f_girth", "side_r_girth", "rear_width", "cow_pixels", "sticker_pixels"]
target = "weight"

# Split data into training and testing sets (assuming data is already loaded)
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r_square = r2_score(y_test, y_pred)
print(r_square)

def save_uploaded_file(uploaded_file, filename):
    """
  Saves a uploaded file to a directory.

  Args:
      uploaded_file: A Streamlit file uploader object.
      filename: The desired filename for the saved file.

  Returns:
      The path of the saved file.
  """
    if uploaded_file is not None:
        # Create a directory named "uploaded_images" if it doesn't exist
        if not os.path.exists("uploaded_images"):
            os.makedirs("uploaded_images")
        filepath = os.path.join("uploaded_images", filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())
        return filepath
    else:
        return None


def predict_weight(side_img_path, rear_img_path):
    """
  Predicts weight based on image paths and returns predicted weight.

  Args:
      side_img_path: Path to the side image.
      rear_img_path: Path to the rear image.

  Returns:
      Predicted weight.
  """
    kpt = cattle_inference.infer_keypoints(side_img_path, rear_img_path)
    side_kpt = kpt[0][0].tolist()
    rear_kpt = kpt[1][0].tolist()
    args = cattle_inference.return_args(side_kpt, rear_kpt)
    pixels = cattle_inference.return_pixels(side_img_path, cattle_inference.side_segmentation_model)
    args = args + pixels

    new_data = {}
    for index, i in enumerate(features):
        new_data[i] = [args[index]]
    new_df = pd.DataFrame(new_data)

    predicted_weight = model.predict(new_df)[0] / 2
    # gradient boosting to be explored, or some other model like polynomial regression
    return predicted_weight


st.title("Cattle Weight Prediction")

# Upload side and rear images using Streamlit
side_image_uploaded = st.file_uploader("Upload Side Image (jpg, jpeg, png)")
rear_image_uploaded = st.file_uploader("Upload Rear Image (jpg, jpeg, png)")

if side_image_uploaded is not None and rear_image_uploaded is not None:
    # Save uploaded images to temporary locations (optional)
    # ... (code to save images if needed)

    # Get image paths (replace with actual saving logic if temporary locations used)

    side_img_path = save_uploaded_file(side_image_uploaded, "Side.jpg")
    rear_img_path = save_uploaded_file(rear_image_uploaded, "Rear.jpg")

    st.image([side_img_path, rear_img_path], caption=["Side Image", "Rear Image"], width=350)
    # st.image(rear_img_path, caption="Rear Image", use_column_width=False)
    # Predict weight
    predicted_weight = predict_weight(side_img_path, rear_img_path)

    # Display predicted weight
    st.success("Prediction Complete!")
    st.write(f"Predicted Weight: {predicted_weight}")
else:
    st.warning("Please upload both side and rear images.")
