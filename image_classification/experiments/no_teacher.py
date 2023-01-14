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

args = get_args(description='No Teacher', mode='train')
expt = 'no-teacher'


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
    "num_epochs": args.epoch,
    "learning_rate": 1e-4 if args.learning_rate is None else args.learning_rate,
    "seed": args.seed,
    "percentage":args.percentage,
    "gpu": args.gpu,
    "experiment": "No Teacher",
    'teacher_training' : args.teacher_training,
    'no_pre_trained_teacher' : args.no_pre_trained_teacher,
}

data = get_dataset(dataset=hyper_params['dataset'],
                   batch_size=hyper_params['batch_size'],
                   percentage=args.percentage)
if hyper_params['teacher_training']:
    if hyper_params['model'].startswith('resnet34'):
        net = vision_models.resnet34(pretrained=hyper_params['no_pre_trained_teacher'])
    elif hyper_params['model'].startswith('resnet50'):
        net = vision_models.resnet50(pretrained=hyper_params['no_pre_trained_teacher'])
    elif hyper_params['model'].startswith('resnet101'):
        net = vision_models.resnet101(pretrained=hyper_params['no_pre_trained_teacher'])
    elif hyper_params['model'].startswith('alexnet'):
        net = vision_models.alexnet(pretrained=hyper_params['no_pre_trained_teacher'])
    if hyper_params['model'].startswith('resnet'):
        fc_in_features = net.fc.in_features
        net.fc = nn.Linear(fc_in_features, 10)
    elif hyper_params['model'].startswith('alex'):
        for param in net.parameters():
            param.requires_grad = False
        fc_in_features=net.classifier[6].in_features
        net.classifier[6] = nn.Linear(fc_in_features, 10)
        for param in net.classifier.parameters():
            param.requires_grad=True
    
## Rajid loaded below a student, but we don't use it anymore...
else:
    net = get_model(hyper_params['model'], hyper_params['dataset'])
net = net.to(args.gpu)

if args.api_key:
    project_name = expt + '-' + hyper_params['model'] + '-' + hyper_params['dataset']
    experiment = Experiment(api_key=args.api_key, project_name=project_name, workspace=args.workspace)
    experiment.log_parameters(hyper_params)

optimizer = torch.optim.Adam(net.parameters(), lr=hyper_params["learning_rate"])
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

