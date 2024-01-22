import os
from datetime import datetime
import cv2
import face_recognition


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tString = time_now.strftime('%H:%M:%S')
            dString = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tString},{dString}')


# Load the photo for face recognition
photo_path = 'photo3.jpg'
input_image = cv2.imread(photo_path)

# Load images and encode faces from the 'Images_Attendance' directory (similar to the original script)
path = 'Images_Attendance'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

encodeListKnown = []
for img in images:
    img_encoding = face_recognition.face_encodings(img)[0]
    encodeListKnown.append(img_encoding)

# Find faces in the input photo and compare them with the encoded faces
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
face_locations = face_recognition.face_locations(input_image_rgb)
face_encodings = face_recognition.face_encodings(input_image_rgb, face_locations)

for face_encoding, face_location in zip(face_encodings, face_locations):
    matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
    name = "Unknown"

    # If a match is found, use the name from classNames list
    if True in matches:
        first_match_index = matches.index(True)
        name = classNames[first_match_index]
        markAttendance(name)

    y1, x2, y2, x1 = face_location
    cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 250, 0), 2)
    cv2.rectangle(input_image, (x1, y2 - 35), (x2, y2), (0, 250, 0), cv2.FILLED)
    cv2.putText(input_image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

# Display the output image with recognized faces
cv2.imshow('Recognized Faces', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
