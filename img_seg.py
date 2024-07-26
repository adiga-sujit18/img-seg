import streamlit as st
import torch
from PIL import Image
import os
import numpy as np
import cv2
import torch.nn as nn
from io import BytesIO
import base64

# UNet Model
# Building double convolution layer
class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
    self.bn1 = nn.BatchNorm2d(out_channels)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.relu = nn.ReLU()

  def forward(self, x):         # Here x is the input to that encoder block
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    return x

# Building Encoder block
class EncoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.conv = DoubleConv(in_channels, out_channels)
    self.pool = nn.MaxPool2d(2)

  def forward(self, inputs):
    x = self.conv(inputs)
    p = self.pool(x)

    return x, p

# Building Bottle Neck
class BottleNeck(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x):
    x = self.conv(x)

    return x

# Building Decoder block
class DecoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0)
    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, input, skip):
    x = self.up(input)
    x = torch.cat([x, skip], axis = 1)
    x = self.conv(x)

    return x

class UNet(nn.Module):
  def __init__(self):
    super().__init__()

    # """ Encoder """
    self.e1 = EncoderBlock(3, 64)
    self.e2 = EncoderBlock(64, 128)
    self.e3 = EncoderBlock(128, 256)
    self.e4 = EncoderBlock(256, 512)

    # """ BottleNeck """
    self.b = BottleNeck(512, 1024)

    # """ Decoder """
    self.d1 = DecoderBlock(1024, 512)
    self.d2 = DecoderBlock(512, 256)
    self.d3 = DecoderBlock(256, 128)
    self.d4 = DecoderBlock(128, 64)

    # """ Classifier """
    self.classifier = nn.Conv2d(64, 1, kernel_size = 1, padding = 0)

  def forward(self, input):
    # """ Encoder """
    e1, p1 = self.e1(input)
    e2, p2 = self.e2(p1)
    e3, p3 = self.e3(p2)
    e4, p4 = self.e4(p3)

    # """ BottleNeck """
    b = self.b(p4)

    # """ Decoder """
    d1 = self.d1(b, e4)
    d2 = self.d2(d1, e3)
    d3 = self.d3(d2, e2)
    d4 = self.d4(d3, e1)

    # """ Classifier """
    x = self.classifier(d4)

    return x

# Function to load the model
def load_model(path, map_location):
    # Load the checkpoing through .pth file
    checkpoint = torch.load(path, map_location)

    # Instantiate the model
    model = UNet()
    # Load the model to the device (gpu/cuda or cpu)
    model = model.to(map_location)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    return model

# Function to parse the mask
def mask_parse(mask):
    # Converting 2D arrays to 3D arrays
    # (512, 512) -> (512, 512, 1)
    mask = np.expand_dims(mask, axis=-1)
    # (512, 512, 1) -> (512, 512, 3)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    return mask

# Function to preprocess Image
def preprocess_image(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read the image as array from the specified path
    image_arr = cv2.imread(path, cv2.IMREAD_COLOR)

    # Crop the image to 512x512
    image_arr = cv2.resize(image_arr, (512, 512))

    # (512, 512, 3) -> (3, 512, 512) ---- torch requirements
    image_arr = np.transpose(image_arr, (2, 0, 1))

    # Normalize the array 
    image_arr = image_arr / 255.0

    # Add an extra dimension: (3, 512, 512) -> (1, 3, 512, 512)
    image_arr = np.expand_dims(image_arr, axis=0)

    # Convert the data type of array from float64 to float32
    image_arr = image_arr.astype(np.float32)

    # Convert numpy.ndarray to torch.tensor
    image_arr = torch.from_numpy(image_arr)

    # Load the array to device (gpu/cuda or cpu)
    image_arr = image_arr.to(device)

    return image_arr

# Function to postprocess Mask
def postprocess_mask(image):
    # if 'model' not in st.session_state:
    #   # Load and store the model in session state
    #   model = load_model(path = 'checkpoint_1.pth', map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #   st.session_state.model = model
  
    # model = st.session_state['model']

    with torch.no_grad():
      # Model called and a new tensor generated
      pred_y = model(image)

      # Sigmoid function applied on the tensor
      pred_y = torch.sigmoid(pred_y)

      # The tensor is converted to a numpy array
      pred_y = pred_y[0].cpu().numpy()

      # (1, 512, 512) -> (512, 512)
      pred_y = np.squeeze(pred_y, axis=0)

      # Convert all values to bool
      pred_y = pred_y > 0.5
  
      # Get a binary array (with 0s and 1s)
      pred_y = np.array(pred_y, dtype = np.uint8)

    return pred_y

# Function to navigate to a new page
def navigate_to(page):
    st.session_state['page'] = page

# Decoding the encoded base64 string
def base64_webp_to_png(base64_string):
    # Convert base64 encoded string to bytes
    data = base64.b64decode(base64_string)

    # Read the bytes
    image = BytesIO(data)

    # Convert bytes into numpy array
    with Image.open(image) as img:
       img_array = np.array(img)

    # Numpy array to image
    img = Image.fromarray(img_array)

    return img

# The the query parameter (base64 encoded string)
img_query = st.query_params.get('img_str')

if img_query is not None:

  # Replace special characters into their usual characters
  img_query = img_query.replace('-', '+')
  img_query = img_query.replace('_', '/')
  img_query = img_query.replace('.', '=')

  if 'model' not in st.session_state:
    # Load and store the model in session state
    model = load_model(path = 'checkpoint_1.pth', map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    st.session_state.model = model
  
  model = st.session_state['model']

  image_from_query = base64_webp_to_png(img_query)

  # Crop the image to 512x512
  image_shape = np.array(image_from_query).shape

  image_arr = np.array(image_from_query)
  if image_shape[0] < image_shape[1]:
    new_dim_x_1 = int((image_shape[1] - image_shape[0])/2)
    new_dim_x_2 = int((image_shape[1] + image_shape[0])/2)
    image_arr = image_arr[:, new_dim_x_1:new_dim_x_2, :]

  elif image_shape[0] > image_shape[1]:
    new_dim_y_1 = int((image_shape[0] - image_shape[1])/2)
    new_dim_y_2 = int((image_shape[0] + image_shape[2])/2)
    image_arr = image_arr[new_dim_y_1:new_dim_y_2, :, :]
  
  else:
      image_arr = image_arr
  
  image_arr = cv2.resize(image_arr, (512, 512))
  image = Image.fromarray(image_arr)

  # Save the user input image with a path
  image.save('user_input.png')

  # Preprocess the image to make it suitable for inputting to the model
  preprocessed_image = preprocess_image("user_input.png")

  # Postprocess the model's output to make it into a suitable image
  pred_y = postprocess_mask(preprocessed_image)
  
  # (512, 512) -> (512, 512, 3)
  pred_y = mask_parse(pred_y)
  pred_y = pred_y * 255

  # Convert the array into image
  mask_generated = Image.fromarray(pred_y)

  col1, col2 = st.columns([0.5, 0.5])
  with col1:
    st.image(image = image_from_query, caption='Input image', use_column_width=True)
    
  
  with col2:
     st.image(image = mask_generated, caption='Generated Mask', use_column_width=True)

  os.remove('user_input.png')

if img_query is None:

  # Initialize the session state
  if 'page' not in st.session_state:
      st.session_state['page'] = 'home'

  if st.session_state['page'] == 'home':
      st.title('Medical Image Segmentation')
      st.header('Welcome!')

      st.markdown('''
                  This is a Machine Learning project on the topic :red[Image Segmentation of Medical Imaging].
      ''')

      st.divider()

      st.subheader('Description')
      st.write('This interface is linked with the trained machine learning model of [UNet](https://www.geeksforgeeks.org/u-net-architecture-explained/).')
      st.write('The model is trained on the [Retina Blood Vessel](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel) dataset from Kaggle.')

      st.write('The aim of the model is to segment the blood vessels present in the medical image scan of a retina. The input to the application is a coloured retina scan and it will output a B/W segmented mask corresponding to the input image.')
      st.write('Below is an illustration of the input image and its corresponding output.')
      st.image('demo_img.png', caption='Illustration of the input scan and the corresponding output')

      try_it_out = st.button("Try it out!")

      st.divider()

      # !!WORK NOT COMPLETED!!

      st.header('Contact Details')
      st.write('If you feel interested in discussing more about this project, collaborating or just wish to connect, feel free to reach out to me!')

      # To display images adjacent to each other
      col1, col2, col3 = st.columns([0.25, 0.1, 0.65])

      with col1:
          st.image('Profile_pic.jpeg')

      with col3:
          st.write('Sujit Adiga')
          # st.write('+91 96869 29348')
      
      if try_it_out is not None:
          navigate_to('demo')

  elif st.session_state['page'] == 'demo':
      st.title('Medical Image Segmentation')

      if 'model' not in st.session_state:
          # Load and store the model in session state
          model = load_model('checkpoint_1.pth', map_location = torch.device('cpu'))
          st.session_state.model = model
          st.write("Model loaded successfully!")
      
      model = st.session_state['model']

      # Introduction
      st.header('Blood vessel segmentation of Retina scans')
      st.divider()
      st.write('Welcome to this page!')
      st.markdown('''You can upload an image of a retina scan and this model will generate a mask for the corresponding image.
                  Make sure to read the instructions before uploading the image.''')

      # Instructions to upload the image
      st.subheader('Instructions')
      st.markdown('''
                  Please carefully read the following instructions before uploading the image:
                  1. The image should have either .png, .jpg or .jpeg extensions
                  2. The image should be a coloured image (RBG)
                  ''')

      # Acknowledgement
      acknowledgment = st.checkbox('I have read the above instructions and made sure that the image I am uploading satisfies the above points.')

      # Input of image
      if acknowledgment:
          uploaded_image = st.file_uploader(label='Upload the image of the scan', 
                                            type=['png', 'jpeg', 'jpg'],
                                            key='img_input')
      
          if uploaded_image is not None:
              # Display the uploaded image
              image = Image.open(uploaded_image)

              # Crop the image to 512x512
              image_shape = np.array(image).shape

              image_arr = np.array(image)
              if image_shape[0] < image_shape[1]:
                new_dim_x_1 = int((image_shape[1] - image_shape[0])/2)
                new_dim_x_2 = int((image_shape[1] + image_shape[0])/2)
                image_arr = image_arr[:, new_dim_x_1:new_dim_x_2, :]

              elif image_shape[0] > image_shape[1]:
                new_dim_y_1 = int((image_shape[0] - image_shape[1])/2)
                new_dim_y_2 = int((image_shape[0] + image_shape[2])/2)
                image_arr = image_arr[new_dim_y_1:new_dim_y_2, :, :]
              
              else:
                 image_arr = image_arr
              
              image_arr = cv2.resize(image_arr, (512, 512))
              image = Image.fromarray(image_arr)
              

              # Center align the image
              col1, col2, col3 = st.columns([0.13, 0.6, 0.27])
              with col2:
                 st.image(image, caption='Image of the Retina Scan', use_column_width=False)

              st.write('Please click on next if the image you have uploaded seems fine.')

              # To create a popover
              with st.popover("Next", use_container_width=True):
                # Define the directory to save the image
                save_dir = "uploaded_images"
                os.makedirs(save_dir, exist_ok=True)

                # Define the save path for the image
                save_path = os.path.join(save_dir, uploaded_image.name)

                # Save the image to the defined path
                image.save(save_path)

                # Preprocess the image to make it suitable for inputting to the model
                preprocessed_image = preprocess_image(save_path)

                # Postprocess the model's output to make it into a suitable image
                pred_y = postprocess_mask(preprocessed_image)
                
                # (512, 512) -> (512, 512, 3)
                pred_y = mask_parse(pred_y)
                pred_y = pred_y * 255
                mask_generated = Image.fromarray(pred_y)

                # To display images adjacent to each other
                col1, col2 = st.columns([0.5, 0.5])
                
                with col1:
                  st.image(image = image, caption='Input image', use_column_width=True)

                with col2:
                  st.image(image = mask_generated, caption='Generated Mask', use_column_width=True)

                os.remove(save_path)



