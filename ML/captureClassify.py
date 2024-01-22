import torch
from torchvision import transforms
from PIL import Image
import cv2
import serial
from trashclassifier import TrashClassifier
import time

modelpath = r'C:\Users\anjan\Downloads\finetunedClassifier.pth'

classifier = TrashClassifier()
classifier.load_state_dict(torch.load(modelpath))
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.7422, 0.7366, 0.7117], [0.3060, 0.3009, 0.3302])
])

ser = serial.Serial('COM5', 9600, 8)

def capture_and_process_photo():
    returnVal, photo = cam.read();

    #convert image to PIL
    capture = Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
    input_image = transform(capture)
    input_batch = input_image.unsqueeze(0)

    with torch.no_grad():
        output = classifier(input_batch)
        print(f'output: {output.size(1)}')
    _, predicted_class = output.max(1)

    #waste categories: electronic, organic, recycle, trash
    class_names = ['e', 'o', 'r', 't']
    predicted_class_name = class_names[predicted_class[0].item()]

    print(f'The predicted class is {predicted_class_name}')
    ser.write(predicted_class_name.encode())
    print('num: {predicted_class_name}')
    

cam = cv2.VideoCapture(0)
cv2.namedWindow("tsest")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test",frame)
    key = cv2.waitKey(1)
    if key == ord('j'):
        capture_and_process_photo()

    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
ser.close()
