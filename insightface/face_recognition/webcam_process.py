from face_detection import face_detect
import os

def createFolder(path, name):
    index = ''
    while True:
        try:
            file_path = os.path.join(path, name+index)
            os.makedirs(file_path)
            return file_path
        except:
            if index:
                index = '('+str(int(index[1:-1])+1)+')' # Append 1 to number in brackets
            else:
                index =  '(1)'
            pass # Go and try create file again

os.makedirs('faces/tmp', exist_ok=True)
os.makedirs(os.path.join('faces/tmp','other'), exist_ok=True)
name = input("Dien ten: ")
name = name.lower()
name = name.replace(" ", "")
file_path = createFolder('faces/tmp',name)
face_detect(file_path, name)

