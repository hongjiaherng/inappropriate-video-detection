def add_model_args(parser):
    parser.add_argument("--svm", type=int, help="maximum number of epochs to train")
    parser.add_argument("--svm2", type=int, help="number of instances, i.e., number of videos in a batch of data")

    raise NotImplementedError()
