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
from torchvision.models import resnet50

args = get_args(description='Hinton KD', mode='train')
expt = 'hinton-kd'

torch.manual_seed(args.seed)
if args.gpu != 'cpu':
    args.gpu = int(args.gpu)
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

hyper_params = {
    "dataset": args.dataset,
    "model": args.model,
    "num_classes": 10,
    "batch_size": 64,
    "num_epochs": args.epoch,
    "learning_rate": 1e-3 if args.learning_rate is None else args.learning_rate,
    "momentum": 0.9,
    "seed": args.seed,
    "percentage":args.percentage,
    "gpu": args.gpu,
    "temperature" : 20,
    "alpha" : 0.2,
    "weight_decay": 5e-4,
    "stage":0,
    "experiment": "Hinton",
    "teacher": args.teacher,
    "teachers_num": args.teachers_num ,
    "teacher_models": args.teacher_models,
    "update_teacher": args.update_teacher,
    "teacher_training": args.teacher_training,
    "save_trained_teacher_model": args.save_model
}

data = get_dataset(dataset=hyper_params['dataset'],
                   batch_size=hyper_params['batch_size'],
                   percentage=args.percentage)

savename = get_savename(hyper_params, experiment=expt)

learn, net = get_model(hyper_params['model'], hyper_params['dataset'], data, teach=True)
learn.model, net = learn.model.to(args.gpu), net.to(args.gpu)

## Should make it a utility function
# multi teacher
if hyper_params['teachers_num'] and hyper_params['teachers_num']>1:
    assert hyper_params['teacher_models'] is not None
    teachers_list = load_teachers_list(hyper_params['teacher_models'], update_teacher=hyper_params['update_teacher'])
# single teacher
elif hyper_params['teacher'] and hyper_params['teachers_num'] is None:
    teachers_list = [load_teacher(hyper_params['teacher'], hyper_params['update_teacher'])]
    # teacher = resnet50()
    # fc_in_features = teacher.fc.in_features
    # teacher.fc = nn.Linear(fc_in_features, 10)
    # teacher.load_state_dict(torch.load('saved_models/imagewoof/full_data/no-teacher/resnet50_classifier/model0.pt'))
    # for param in teacher.parameters():
    #     param.requires_grad = False
    # print(teacher)
    # teacher.load_state_dict(torch.load('saved_models/imagewoof/full_data/no-teacher/resnet50_classifier/model42.pt'))
    # print("### Worked to load pre-trained resnet-50!! ###")
else:
    teachers_list = [learn.model]

sf_student = None
sf_teacher = None

if args.api_key:
    project_name = expt + '-' + hyper_params['model'] + '-' + hyper_params['dataset']
    experiment = Experiment(api_key=args.api_key, project_name=project_name, workspace=args.workspace)
    experiment.log_parameters(hyper_params)

# optimizer = torch.optim.SGD(net.parameters(), lr=hyper_params["learning_rate"], momentum=hyper_params["momentum"], weight_decay=hyper_params["weight_decay"])
optimizer = torch.optim.Adam(net.parameters(), lr=hyper_params["learning_rate"])

loss_function = nn.KLDivLoss(reduction='mean')
loss_function2 = nn.CrossEntropyLoss()
best_val_loss = 100

for epoch in range(hyper_params["num_epochs"]):
    net, train_loss, val_loss, _, best_val_loss = train(net,
                                                        teachers_list,
                                                        data,
                                                        sf_teacher,
                                                        sf_student,
                                                        loss_function,
                                                        loss_function2,
                                                        optimizer=optimizer,
                                                        hyper_params=hyper_params,
                                                        epoch=epoch,
                                                        savename=savename,
                                                        best_val_acc=best_val_loss,
                                                        expt=expt
                                                        )
    if args.api_key:
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metric("val_loss", val_loss)

