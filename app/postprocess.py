def postprocess(face_detector, scene, markers_json=None, *args, **kwargs):

    from app import SessionReader
    from app.utils import create_learning_dataset

    sess_reader = SessionReader()
    sess_reader.fit('1531844043', r'D:\param_train_sess\17_07_18\1531844043', cams=scene.cams, by='basler')

    markers = []
    for i in range(3):
        for j in range(8):
            marker = markers_json.get(f'wall_{i}_dot_{j+1}')
            if marker:
                markers.extend([marker] * 100)

    create_learning_dataset('../brs/',
                            sess_reader,
                            face_detector,
                            scene,
                            indices=range(len(sess_reader.snapshots)),
                            markers=markers)
