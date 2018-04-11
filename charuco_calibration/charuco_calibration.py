"""
Charuco calibration

"""
import cv2

class Charuco_calibration:
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.board = cv2.aruco.CharucoBoard_create(5, 7, 0.0725, 0.0425, self.dictionary)
        
    def calibrate(self):
        capture = cv2.VideoCapture(cv2.CAP_ANY)
        allCorners = []
        allIds = []
        decimator = 0
        
        while capture.isOpened():
            ret,frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.aruco.detectMarkers(gray, self.dictionary)
            if len(res[0])>0:
                res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray, self.board)
                if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%3==0:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])
                    cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
                    cv2.aruco.drawAxis()
            cv2.imshow('frame',gray)
            if cv2.waitKey(33) % 256 == 27:
                print('ESC pressed, closing...')
                break
            decimator+=1
    
        capture.release()
        cv2.destroyAllWindows()
            
if __name__ == '__main__':
    char_calibr = Charuco_calibration()
    char_calibr.calibrate()

