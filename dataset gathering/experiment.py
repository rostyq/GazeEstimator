def run_experiment(model = 'tutorial'):
    cap = cv2.VideoCapture(0)
    root = tk.Tk()
    screen = root.winfo_screenheight(), root.winfo_screenwidth()

    # Display the resulting frame
    cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    #init face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    i = 0
    while(True):
        ret, frame = cap.read()
        key = cv2.waitKey(1)

        if key == 27:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #faces detection
        rects = detector(gray, 1)
        if len(rects) > 0:
            _, frame = get_face_pose(rects, gray, frame, predictor, model)

        clear_output(wait=True)
        cv2.imshow('test', frame)

    # When everything done, release the capture    
    cap.release()
    cv2.destroyAllWindows()