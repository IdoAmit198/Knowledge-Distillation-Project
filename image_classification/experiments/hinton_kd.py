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
from torchvision.models import resnet50, resnet34, resnet18
from mutual_trainer_new import mutual_train
from epoch_trainer import epoch_train

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
    "teacher_training": args.teacher_training
}

print()

# We ended up with multiple teachers of the same architecture (e.g resnet34)
# But we need a way to group in wandb by architecture, so I add this attribute.
if hyper_params['teacher']:
    if hyper_params['teacher'].startswith('resnet34'):
        hyper_params['teacher_architecture'] = 'resnet34'
    elif hyper_params['teacher'].startswith('resnet50'):
        hyper_params['teacher_architecture'] = 'resnet50'
    elif hyper_params['teacher'].startswith('resnet101'):
        hyper_params['teacher_architecture'] = 'resnet101'
    elif hyper_params['teacher'].startswith('vit'):
        hyper_params['teacher_architecture'] = 'vit'

data = get_dataset(dataset=hyper_params['dataset'],
                   batch_size=hyper_params['batch_size'],
                   percentage=args.percentage)

savename = get_savename(hyper_params, experiment=expt)

learn, net = get_model(hyper_params['model'], hyper_params['dataset'], data, teach=True)
learn.model, net = learn.model.to(args.gpu), net.to(args.gpu)

## Should make it a utility function
if hyper_params['teachers_num'] and hyper_params['teachers_num']>1:
    assert hyper_params['teacher_models'] is not None
    teachers_list = load_teachers_list(hyper_params['teacher_models'], update_teacher=hyper_params['update_teacher'])
elif hyper_params['teacher'] and hyper_params['teachers_num'] is None:
    teachers_list = [load_teacher(hyper_params['teacher'], hyper_params['update_teacher'])]
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

loss_function = nn.KLDivLoss(reduction='batchmean')
# loss_function = nn.CrossEntropyLoss()
loss_function2 = nn.CrossEntropyLoss()
best_val_loss = 100

if args.mutual_learning is None:
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

else:
    optimizers = []
    mutual_nets = []
    best_val_loss_list = []
    for k in range(args.mutual_learning):
        # if k == 0:
        #     mutual_net = resnet34(pretrained=False)
        # else:
        #     mutual_net = resnet18(pretrained=False)
        mutual_net = resnet18(pretrained=False)
        # mutual_net = resnet34(pretrained=False)
        # mutual_net = get_model(hyper_params['model'], hyper_params['dataset'], data, teach=False)
        mutual_net = mutual_net.to(args.gpu)
        print(f'mutual_net before list id is: {id(mutual_net)}')
        mutual_nets.append(mutual_net)
        print(f'mutual_nets before list id is: {id(mutual_nets[k])}')
        mut_optimizer = torch.optim.Adam(mutual_net.parameters(), lr=hyper_params["learning_rate"])
        print(f'optimizer before list id is: {id(mut_optimizer)}')
        optimizers.append(mut_optimizer)
        print(f'optimizers before list id is: {id(optimizers[k])}')
        best_val_loss_list.append(100)

    for epoch in range(hyper_params["num_epochs"]):
        for i in range(args.mutual_learning):
            student = mutual_nets[i]
            for param in student.parameters():
                param.requires_grad = True

            if epoch < args.num_epochs_training_separately:
                teachers_list = None
            else:
                teachers_list = [mutual_nets[j] for j in range(args.mutual_learning) if j != i]
                for teacher in teachers_list:
                    for param in teacher.parameters():
                        param.requires_grad = False

            student, train_loss, val_loss, _, best_val_loss_list[i] = epoch_train(mutual_nets[i],
                                                                        teachers_list,
                                                                        data,
                                                                        sf_teacher,
                                                                        sf_student,
                                                                        loss_function,
                                                                        loss_function2,
                                                                        optimizer=optimizers[i],
                                                                        hyper_params=hyper_params,
                                                                        epoch=epoch,
                                                                        savename=savename,
                                                                        best_val_acc=best_val_loss_list[i],
                                                                        expt=expt,
                                                                        num_epochs_training_separately = args.num_epochs_training_separately
                                                                        )


        # # optimizer0 = torch.optim.Adam(mutual_nets[0].parameters(), lr=hyper_params["learning_rate"])
        # # optimizer1 = torch.optim.Adam(mutual_nets[1].parameters(), lr=hyper_params["learning_rate"])
        # student, train_loss, val_loss, _ = mutual_train(mutual_nets,
        #                                                     data,
        #                                                     loss_function,
        #                                                     loss_function2,
        #                                                     optimizers,
        #                                                     hyper_params=hyper_params,
        #                                                     epoch=epoch,
        #                                                     savename=savename,
        #                                                     expt=expt,
        #                                                     num_epochs_training_separately = args.num_epochs_training_separately
        #                                                     )

        # if args.api_key:
        #     experiment.log_metric(f"train_loss_{k}", train_loss)
        #     experiment.log_metric(f"val_loss_{k}", val_loss)

