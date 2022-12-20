import cv2
import joblib
import streamlit as st
from PIL import Image
from keras_preprocessing.image import load_img,img_to_array
from io import BytesIO
import base64

detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2022mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)
detector.setInputSize((320, 320))

recognizer = cv2.FaceRecognizerSF.create(
            "face_recognition_sface_2021dec.onnx","")
svc = joblib.load('svc.pkl')
mydict = ['Ban Anh', 'Ban Bao','Ban Dat', 'Ban Dien', 'Ban Ky', 'Ban Manh', 'Ban Nam', 'Ban Ninh', 'Ban Sang', 'Ban Thanh', 'Ban Tuan', 'Duc Hoa', 'Le Tai', 'BanSon', 'Song Huy', 'Thay Duc']


def onRecognition(img_path):
        img=load_img(img_path,target_size=(320,320,3))
        imgin=img_to_array(img)
        faces = detector.detect(imgin)
        face_align = recognizer.alignCrop(imgin, faces[1][0])
        face_feature = recognizer.feature(face_align)
        test_prediction = svc.predict(face_feature)
        result = mydict[test_prediction[0]]
        return result

def run():


    st.markdown("<h1 style='text-align: Left; color: #E67E22;'>Face Detector</h1>", unsafe_allow_html=True)
    st.write(
        ":dog: This is a website to detect human face. You can try ! " )
    st.sidebar.markdown("<h1 style='text-align: Center; color: #E67E22;'>Digital Image Processing</h1>", unsafe_allow_html=True)
    st.sidebar.write("## Upload :gear:")
    img_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png","bmp"])
    if img_file is not None:
        img = Image.open(img_file).resize((320,320))
        st.image(img,use_column_width=False)
        
        save_image_path = '../upload/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        if st.button("Recognition"):
            if img_file is not None:
                result= onRecognition(save_image_path)
                st.success("**Recognition : "+result+'**')
                st.success('Detect success !', icon="✅")
                txt ="Recognition :" + result
run()