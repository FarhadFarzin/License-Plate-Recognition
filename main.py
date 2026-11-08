import cv2
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



cv2.imshow('Orginal',img)
cv2.waitKey(0)
cv2.destroyAllWindows()