class hparams:

    train_or_test = 'test'
    output_dir = 'logs/googleNet'
    aug = False
    latest_checkpoint_file = 'checkpoint_latest.pt'
    total_epochs = 5
    batch_size = 8
    ckpt = 'best_model.pth'
    init_lr = 0.0002
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    mode = '2d' # '2d or '3d'
    in_class = 1
    out_class = 2

    crop_or_pad_size = 224,224,1 # if 3D: 256,256,256

    fold_arch = '*.png'


