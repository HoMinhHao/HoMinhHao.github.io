import os
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from deepface import DeepFace

models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
  "GhostFaceNet",
]

backends = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'fastmtcnn',
  'retinaface',
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

metrics = ["cosine", "euclidean", "euclidean_l2"]

alignment_modes = [True, False]

original_img_dir_path = '/content/drive/MyDrive/DeepFace/Original_Images' # Replace with your original images directory path
face_cropped_dir_path = '/content/drive/MyDrive/DeepFace/Face_images' # Replace with your targer faces extracted directory path

files = os.listdir(original_img_dir_path)
for file in files:
  path=os.path.join(face_cropped_dir_path,file)
  os.makedirs(path, exist_ok=True)

def crop_face(original_img_dir_path, face_cropped_dir_path):
  dirs = os.listdir(original_img_dir_path)
  for dir in dirs:
    files=os.path.join(original_img_dir_path,dir)

    for file in os.listdir(files):
      img_path=os.path.join(files,file)

      save_img_path = os.path.join(face_cropped_dir_path, dir, file)

      img=cv2.imread(img_path)
      face = DeepFace.extract_faces(
        img_path = img_path,
        detector_backend = backends[7],
        align = alignment_modes[0],
      )
      face_coordinates = face[0]["facial_area"]
      x, y, w, h = face_coordinates["x"], face_coordinates["y"], face_coordinates["w"], face_coordinates["h"]

      image = Image.open(img_path)

      # Define the coordinates for cropping (left, upper, right, lower)
      left = x
      top = y
      right = x+w
      bottom = y+h

      # Crop the image
      cropped_image = image.crop((left, top, right, bottom))

      # Save the cropped image
      cropped_image.save(save_img_path)
      
if __name__ == '__main__':
    crop_face(original_img_dir_path, face_cropped_dir_path)
