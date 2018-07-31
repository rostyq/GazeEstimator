def gather(dataset_path, face_detector, scene, person_name, dataset_size, size='_72_120'):

    from app.utils import experiment_without_BRS

    experiment_without_BRS(dataset_path,
                           face_detector,
                           scene,
                           person_name,
                           size=size,
                           dataset_size=dataset_size)
