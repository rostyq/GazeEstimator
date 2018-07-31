def visualize(face_detector, scene, path_to_model, *args, **kwargs):

    from cv2 import imread
    from app.utils import visualize_predict

    back = imread('../screen1.png')
    visualize_predict(face_detector, scene, path_to_model=path_to_model, back=back)
