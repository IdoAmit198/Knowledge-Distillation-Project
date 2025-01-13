from fastai.vision import *
from timm.data.dataset import ImageDataset
from timm.data.loader import create_loader
from timm.data import resolve_data_config, create_transform, create_dataset

def create_datasets(config, train_path, val_path):
    """
    Utility funtion to create datasets equiped with the required augmentations for timm models.
    """
    train_transforms = create_transform(
        **config,
        is_training=True,
    )

    eval_transforms = create_transform(
        **config,
        is_training=False,
    )
    train_dataset = create_dataset(name='', root=train_path, transform=train_transforms)
    eval_dataset = create_dataset(name='', root=val_path, transform=eval_transforms)

    return train_dataset, eval_dataset

def get_dataset(dataset, batch_size, percentage=None, timm_config=None):
    val = 'val'
    sz = 224
    stats = imagenet_stats
    if dataset == 'imagenette' : 
        path = untar_data('https://s3.amazonaws.com/fast-ai-imageclas/imagenette')
        classes = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
    elif dataset == 'cifar10' : 
        path = untar_data(URLs.CIFAR)
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == 'imagewoof' : 
        path = untar_data('https://s3.amazonaws.com/fast-ai-imageclas/imagewoof')
        classes = ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601', 'n02105641', 'n02111889', 'n02115641']
    else:
        sys.exit(f'invalid dataset : {dataset}')
    
    if percentage is not None:
        path = path/('new' + str(percentage))


    train_path = str(path) + '/train'
    val_path = str(path) + '/val'

    tfms = get_transforms(do_flip=False)
    if dataset == 'cifar10' : 
        val = 'test'
        sz = 32
        stats = cifar_stats
    

    if timm_config:
        # print("*" *20 + "Made it in vit_config" + "*"*20)
        train_dataset, eval_dataset = create_datasets(
            config = timm_config,
            train_path=train_path,
            val_path=val_path,
        )
        # print(f"len(train_dataset) = {len(train_dataset)}")
        # print(f"len(eval_dataset) = {len(eval_dataset)}")
        train_dl = create_loader(train_dataset, timm_config['input_size'], batch_size)
        valid_dl = create_loader(eval_dataset, timm_config['input_size'], batch_size)
        return {'train_dl':train_dl, 'valid_dl':valid_dl}

    # print("*" *20 + "Didn't make it in vit_config" + "*"*20)
    return ImageDataBunch.from_folder(path, train='train', valid=val, bs=batch_size, size=sz, ds_tfms=tfms, classes=classes).normalize(stats)
