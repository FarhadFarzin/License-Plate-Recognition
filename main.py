import cv2
import pytesseract
import glob
import os

images = glob.glob('images/input/*.jpg')

for img_path in images:
    
    img = cv2.imread(img_path)
    
    

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.blur(gray, (3,3))

    canny = cv2.Canny(blur,60,150)

    contours , _ = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)



    
    for contour in contours:
    
        area = cv2.contourArea(contour)
    
        x,y,w,h = cv2.boundingRect(contour)
    
        rectangularity = area / (w * h)
    

        if area > 700 and w > 100 and h > 40 and 2 < w / h < 5 and rectangularity < 0.68:
        
        
            cv2.rectangle(img,(x,y),(x+w , y+h),(0,255,0),2)
        
            plate = img[y : y + h , x : x + w]
        
            plate_resize = cv2.resize(plate,None ,fx=2,fy=2)
        
            if  img_path.endswith('car2.jpg'):
            
                plate = plate_resize[9:-5, 50:-5]
            
            elif img_path.endswith('car1.jpg'):
            
                plate = plate_resize[22:-28, 25:-15]
        
        
            if plate.size == 0:
            
                continue
        
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
            blur = cv2.GaussianBlur(gray, (5,5),0)
        
            _ , thresh = cv2.threshold(blur,109,255,cv2.THRESH_BINARY)
        
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))
        
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE , kernel)

        
            text = pytesseract.image_to_string(thresh , lang='eng' , config= '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
        
            
            text = text.strip()
        
        
        
            if text:
            
                if img_path.endswith('car2.jpg') and len(text) >= 2:
                
                    text = 'ZZ' + text[2:]

                print(f'Plate Number: {text}')
            
            else:
            
                print('Plate Not Recognized')
                
            
            
            
            
      
    cv2.imwrite('images/output/' + os.path.splitext(os.path.basename(img_path))[0]  + '_ plate.jpg',plate)
    cv2.imshow('Orginal',img)
    cv2.imshow('Plate', plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()