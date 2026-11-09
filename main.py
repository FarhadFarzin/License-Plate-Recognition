import cv2
import pytesseract

img = cv2.imread('D:/Python_Projects/license_plate_recognition/images/input/car.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray, (3,3))
canny = cv2.Canny(blur,60,150)
contours , _ = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    
    area = cv2.contourArea(contour)
    
    x,y,w,h = cv2.boundingRect(contour)
    
    if area > 800 and 2 < w / h < 5 :
        
        cv2.rectangle(img,(x,y),(x+w , y+h),(0,255,0),2)
        
        plate = img[y : y + h , x : x + w]
        
        plate_resize = cv2.resize(plate,None ,fx=2,fy=2)
        
        plate = plate_resize[22:-28, 15:-15]
        
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray, (5,5),0)
        
        _ , thresh = cv2.threshold(blur,120,255,cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE , kernel)
        
        text = pytesseract.image_to_string(thresh , lang='eng' , config= '--psm 7')
        
        print(text)
        
      
       
        
        



cv2.imshow('Orginal',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()