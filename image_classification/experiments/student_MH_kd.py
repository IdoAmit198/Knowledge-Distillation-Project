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
from trainer_mh import *
import torchvision.models as vision_models


args = get_args(description='Student multi head KD', mode='train')
expt = 'student_MH_kd'

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
    "experiment": "student_MH_kd",
    "teacher": args.teacher,
    "teacher_models": args.teacher_models,
    "update_teacher": args.update_teacher,
    "teacher_training": args.teacher_training,
    "mh_fc_student": args.multi_head_fc_student,
    "aggregate_teachers": args.aggregate_teachers,
}

data = get_dataset(dataset=hyper_params['dataset'],
                   batch_size=hyper_params['batch_size'],
                   percentage=args.percentage)

savename = get_savename(hyper_params, experiment=expt)

## Load a student model, and create the multi-heads.
## For now just loading resnet 18, but should make it a utility function.
## This solution based on 'childern' will work only for Resnet's torch models.
student_embedding = vision_models.resnet18()
# Save the fc input_features and drop the fc head by making it Identity.
fc_in_features = student_embedding.fc.in_features
student_embedding.fc = torch.nn.Identity()

# Load teachers
if hyper_params['teacher_models'] and len(hyper_params['teacher_models'])>=1:
    hyper_params['teachers_num'] = len(hyper_params['teacher_models'])
    teachers_dict = load_teachers_list(hyper_params['teacher_models'], update_teacher=hyper_params['update_teacher'])
elif hyper_params['teacher'] and hyper_params['teachers_num'] is None:
    raise Exception("Provided teachers in \'teacher\' argument, which is deprecated. You should provide it using \'teacher_models\'.")
else:
    raise Exception("ERROR! Provided an empty list of teachers in teacher_models and teacher arguments.")

hyper_params['teacher_architecture'] = get_teachers_architecture(teacher_name=hyper_params['teacher'],\
                                                                teacher_names_list=hyper_params['teacher_models'])

# Consider making it a dict instead of list, with respect to names of teachers and so on.
fc_heads = {'gt_head':torch.nn.Linear(fc_in_features,hyper_params['num_classes'])}
if hyper_params['aggregate_teachers']:
    fc_heads['mean_teachers_head'] = torch.nn.Linear(fc_in_features,hyper_params['num_classes'])
else:
    for teacher_name in hyper_params['teacher_models']:
        fc_heads[f"{teacher_name}_head"] = torch.nn.Linear(fc_in_features,hyper_params['num_classes'])

if args.api_key:
    project_name = expt + '-' + hyper_params['model'] + '-' + hyper_params['dataset']
    experiment = Experiment(api_key=args.api_key, project_name=project_name, workspace=args.workspace)
    experiment.log_parameters(hyper_params)

opt_params = list(student_embedding.parameters())
for head in fc_heads.values():
    opt_params += list(head.parameters())
optimizer = torch.optim.Adam(opt_params, lr=hyper_params["learning_rate"])

loss_function = nn.KLDivLoss(reduction='mean')
# loss_function = nn.CrossEntropyLoss()
loss_function2 = nn.CrossEntropyLoss()
best_val_acc = 10
batches_train_logits_teachers = []
batches_val_logits_teachers = []
all_teachers_samples_certainties_dict = {}
teachers_train_acc_dict = {}
mean_teachers_train_acc = None
teachers_val_acc_dict = {}
mean_teachers_val_acc = None

for epoch in range(hyper_params["num_epochs"]):
    student_embedding, fc_heads, train_loss, val_loss, _, best_val_acc,\
    batches_train_logits_teachers, all_teachers_samples_certainties_dict, \
    teachers_train_acc_dict, mean_teachers_train_acc = train(student_embedding = student_embedding,
                                                        teachers_dict = teachers_dict,
                                                        fc_heads = fc_heads,
                                                        data = data,
                                                        loss_function = loss_function,
                                                        loss_function2 = loss_function2,
                                                        optimizer=optimizer,
                                                        hyper_params=hyper_params,
                                                        epoch=epoch,
                                                        savename=savename,
                                                        best_val_acc=best_val_acc,
                                                        expt=expt,
                                                        batches_train_logits_teachers=batches_train_logits_teachers,
                                                        batches_val_logits_teachers=batches_val_logits_teachers,
                                                        all_teachers_samples_certainties_dict=all_teachers_samples_certainties_dict,
                                                        teachers_train_acc_dict=teachers_train_acc_dict,
                                                        teachers_val_acc_dict=teachers_val_acc_dict,
                                                        mean_teachers_train_acc=mean_teachers_train_acc
                                                        )
    if args.api_key:
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metric("val_loss", val_loss)

