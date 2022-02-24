class FLAGES(object):

    pan_size= 128
    
    ms_size=32
    
    
    num_spectrum=4
    
    ratio=4
    stride=16
    norm=True
    
    
    batch_size=32
    lr=0.0001
    decay_rate=0.99
    decay_step=10000
    
    img_path='./data/source_data'
    data_path='./data/train/train_qk.h5'
    log_dir='./log_11_25-generator'
    model_save_dir='./model_11_25-generator'
    
    is_pretrained=False
    
    iters=100
    model_save_iters = 5
    valid_iters=10
