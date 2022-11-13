import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from comet_ml import Experiment
from fastai.vision import *
import torch
import argparse
import os
from image_classification.arguments import get_args
from image_classification.datasets.dataset import get_dataset
from image_classification.utils.utils import *
from image_classification.models.custom_resnet import *
from trainer import *
import torchvision.models as vision_models
import wandb
from timm.data import resolve_data_config, create_transform
import timm


args = get_args(description='Timm teacher training', mode='train')
expt = 'Timm-teacher-training'

torch.manual_seed(args.seed)
if args.gpu != 'cpu':
    args.gpu = int(args.gpu)
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

hyper_params = {
    "dataset": args.dataset,
    "model": args.model,
    "stage": 0,
    "num_classes": 10,
    "batch_size": 64,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "num_epochs": args.epoch,
    "learning_rate": 1e-4 if args.learning_rate is None else args.learning_rate,
    "seed": args.seed,
    "percentage":args.percentage,
    "gpu": args.gpu,
    "experiment": "Timm Teacher Training",
    'teacher_training' : args.teacher_training
}

config= None
if hyper_params['teacher_training']:
    if hyper_params['model'].startswith('vit'):
        net = timm.models.create_model('vit_small_patch16_224', pretrained=True, num_classes=10)
        # Creating the model specific data transformation
    elif hyper_params['model'].startswith('mobilenet'):
        net = timm.models.create_model('mobilenetv3_small_100', pretrained=True, num_classes=10)
        print("loaded mobilenet model")
    config = resolve_data_config({}, model=net)
    
## Rajid loaded below a student, but we don't use it anymore...
else:
    sys.exit(f"invalid argument. tried to train ViT, but got teacher_training={hyper_params['teacher_training']}")
net = net.to(args.gpu)


data = get_dataset(dataset=hyper_params['dataset'],
                   batch_size=hyper_params['batch_size'],
                   percentage=args.percentage, vit_config=config)

# print("*"*50)
# print(f"data is {data}")

if args.api_key:
    project_name = expt + '-' + hyper_params['model'] + '-' + hyper_params['dataset']
    experiment = Experiment(api_key=args.api_key, project_name=project_name, workspace=args.workspace)
    experiment.log_parameters(hyper_params)

optimizer = torch.optim.SGD(net.parameters(), lr=hyper_params["learning_rate"], momentum=hyper_params["momentum"], weight_decay=hyper_params["weight_decay"])
# optimizer = torch.optim.Adam(net.parameters(), lr=hyper_params["learning_rate"])
loss_function = nn.CrossEntropyLoss()
savename = get_savename(hyper_params, experiment=expt)
best_val_acc = 0
for epoch in range(hyper_params['num_epochs']):
    student, train_loss, val_loss, val_acc, best_val_acc = train(
                                                                student = net,
                                                                teachers_list = None,
                                                                data=data,
                                                                sf_teacher=None,
                                                                sf_student=None,
                                                                loss_function=loss_function,
                                                                loss_function2=None,
                                                                optimizer=optimizer,
                                                                hyper_params=hyper_params,
                                                                epoch=epoch,
                                                                savename=savename,
                                                                best_val_acc=best_val_acc
                                                                )
    

    if args.api_key:
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metric("val_loss", val_loss)
        experiment.log_metric("val_acc", val_acc * 100)

