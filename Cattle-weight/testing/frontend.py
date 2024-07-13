import streamlit as st
import os


def save_uploaded_file(uploaded_file, filename):
    """
  Saves an uploaded file to a directory.

  Args:
      uploaded_file: A Streamlit file uploader object.
      filename: The desired filename for the saved file.

  Returns:
      The path of the saved file.
  """
    if uploaded_file is not None:
        # Create a directory named "uploaded_images" if it doesn't exist
        if not os.path.exists("../deployment/uploaded_images"):
            os.makedirs("../deployment/uploaded_images")
        filepath = os.path.join("../deployment/uploaded_images", filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())
        return filepath
    else:
        return None


st.title("Image Uploader")

# Upload the first image
uploaded_file1 = st.file_uploader("Enter Side Image")
saved_path1 = save_uploaded_file(uploaded_file1, "Side.jpg")

# Upload the second image
uploaded_file2 = st.file_uploader("Enter Rear Image")
saved_path2 = save_uploaded_file(uploaded_file2, "Rear.jpg")

# Display success message if both images uploaded
if saved_path1 and saved_path2:
    st.success("Images uploaded successfully!")
    st.write("Path of image 1:", saved_path1)
    st.write("Path of image 2:", saved_path2)
