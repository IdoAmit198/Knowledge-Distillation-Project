from comet_ml import Experiment
from image_classification.utils import uncertainty_metrics
from tqdm import tqdm

from fastai.vision import *

from image_classification.utils.utils import *
from image_classification.utils import *
import wandb

from datetime import date, datetime
import torch

def mutual_train(mutual_nets, data, loss_function, loss_function2, optimizers, hyper_params, epoch, savename, expt=None, num_epochs_training_separately=0, is_18_before_34 = True):

    for net in mutual_nets:
        print(f'net id is: {id(net)}')
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    today = date.today().strftime("%d/%m")
    models_names = "resnet18-resnet34" if is_18_before_34 else "resnet34-resnet18"
    run = wandb.init(
    project="our-awesome-project",
    entity = "ido-shani-proj" ,
    name= f"Mutual-{len(mutual_nets)}nets-{models_names}-{hyper_params['num_epochs']} epochs-{today}-{current_time}",
    config=hyper_params)
    config = wandb.config
    if hyper_params['teacher_training']:
        artifact = wandb.Artifact(name=hyper_params['model'], type='model')

    if type(data) is dict:
        loop = tqdm(data['train_dl'])
    else:
        loop = tqdm(data.train_dl)
    gpu = hyper_params['gpu']
    cpu = torch.device("cpu")
    
    train_loss_list = [[] for x in range(len(mutual_nets))]
    student_pred_train_list = [[] for x in range(len(mutual_nets))]
    labels_train_list = []
    grads = [{'norm_1': [], 'norm_2': [], 'normalized_norm_1': [], 'normalized_norm_2': []} for x in range(len(mutual_nets))]
    for images, labels in loop:
        if gpu != 'cpu':
            images = torch.autograd.Variable(images).to(gpu).float()
            labels = torch.autograd.Variable(labels).to(gpu)
        else:
            images = torch.autograd.Variable(images).float()
            labels = torch.autograd.Variable(labels)
        
        for i in range(len(mutual_nets)):
            # print(f'student {i} in epoch {epoch}')

            student = mutual_nets[i]
            if (epoch >= num_epochs_training_separately) :
                teachers_list = [mutual_nets[j] for j in range(len(mutual_nets)) if i!=j]
            optimizer = optimizers[i]

            student.train()
            student = student.to(gpu)
            student.zero_grad()

            for param in student.parameters():
                param.grad = torch.zeros_like(param)

            if (epoch >= num_epochs_training_separately):
                for teacher in teachers_list:
                    teacher.eval()
                    teacher = teacher.to(gpu)
                    teacher.zero_grad()

            student_logits = student(images)
            student_pred_train_list[i].append(F.softmax(student_logits, dim = 1))
            if i == 0:
                labels_train_list.append(labels)

            if (epoch >= num_epochs_training_separately):
                if (len(teachers_list) == 9):
                    ten_nets_temp = 0
                    for t, teacher in enumerate(teachers_list):
                        # print(f'teacher number: {t}')
                        with torch.no_grad():
                            teacher_logits = teacher(images)
                        ten_nets_temp = torch.add(ten_nets_temp, teacher_logits)
                    teacher_mean_logits = ten_nets_temp / 9
                else: 
                    teachers_logits_list = []
                    for teacher in teachers_list:
                        teacher_logits = teacher(images)
                        teachers_logits_list.append(teacher_logits)
                    teacher_mean_logits = torch.mean(torch.stack(teachers_logits_list), dim=0)

            TEMP = hyper_params['temperature']
            ALPHA = hyper_params['alpha']
            
            if (epoch >= num_epochs_training_separately) :
                teacher_soft_targets = F.softmax(teacher_mean_logits/TEMP,dim=1)
                student_logits_log_softmax = F.log_softmax(student_logits/TEMP,dim=1)
                distillation_loss = loss_function(student_logits_log_softmax, teacher_soft_targets)*(1-ALPHA)*TEMP*TEMP
                student_loss = loss_function2(student_logits, labels)*(ALPHA)
                # print(f'distillation_loss {distillation_loss}')
                # print(f'student_loss {student_loss}')
                loss = distillation_loss + student_loss
            else :
                loss = loss_function2(student_logits, labels)

            train_loss_list[i].append(loss.item())

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate grad size
            total_norm = {'1': 0, '2': 0}
            for j,p in enumerate(student.parameters()):
                param_size = p.numel()
                total_norm['1'] += p.grad.detach().data.norm(1)/param_size
                total_norm['2'] += p.grad.detach().data.norm(2)/param_size
            batch_grad_1 = total_norm['1']/(j+1)
            batch_grad_2 = total_norm['2']/(j+1)
            grads[i]['norm_1'].append(batch_grad_1)
            grads[i]['norm_2'].append(batch_grad_2)
            grads[i]['normalized_norm_1'].append(batch_grad_1/loss.item())
            grads[i]['normalized_norm_2'].append(batch_grad_2/loss.item())

            loop.set_description('Epoch {}/{}'.format(epoch + 1, hyper_params['num_epochs']))
            loop.set_postfix(loss=loss.item())

    for i in range(len(mutual_nets)) :
        all_student_pred_train = torch.cat(student_pred_train_list[i])
        all_labels_train = torch.cat(labels_train_list)
        samples_certainties = get_samples_certainties(all_student_pred_train, all_labels_train)
        _log_uncertainty("train", samples_certainties, epoch, i)
        
        _, student_train_pred_final = torch.max(all_student_pred_train, 1)
        student_correct = (student_train_pred_final==all_labels_train).sum().item()
        train_acc = student_correct / all_labels_train.size(0)
        
        train_loss = (sum(train_loss_list[i]) / len(train_loss_list[i]))

        student = mutual_nets[i]

        student.eval()
        val_loss_list = list()
        with torch.no_grad():
            correct = 0
            total = 0

            y_pred_val_list = []
            labels_val_list = []
            teacher_pred_val_list = []
            if type(data) is dict:
                valid_loop = data['valid_dl']
            else:
                valid_loop = data.valid_dl
            for _, (images, labels) in enumerate(valid_loop):
                if gpu != 'cpu':
                    images = torch.autograd.Variable(images).to(gpu).float()
                    labels = torch.autograd.Variable(labels).to(gpu)
                else:
                    images = torch.autograd.Variable(images).float()
                    labels = torch.autograd.Variable(labels)

                student_logits = student(images)
                y_pred_val_list.append(F.softmax(student_logits, dim = 1))
                labels_val_list.append(labels)
                
                ALPHA = hyper_params['alpha']

                student_logits = F.log_softmax(student_logits, dim = 1)
                _, pred_ind = torch.max(student_logits, 1)

                total += labels.size(0)
                correct += (pred_ind == labels).sum().item()

                loss = loss_function2(student_logits,labels)

                val_loss_list.append(loss.item())
        
        all_y_pred_val = torch.cat(y_pred_val_list)
        all_labels_val = torch.cat(labels_val_list)

        ##TODO: delete later, only to test accuracy of teacher ResNet34 of paper!!
        samples_certainties = get_samples_certainties(all_y_pred_val, all_labels_val)
        _log_uncertainty("val", samples_certainties, epoch, i)

        val_loss = (sum(val_loss_list) / len(val_loss_list))
        if total > 0:
            val_acc = correct / total
            print(f"## val_acc = val_acc")
        else:
            val_acc = None

        grad_norm_1 = sum(grads[i]['norm_1'])/len(grads[i]['norm_1'])
        grad_norm_2 = sum(grads[i]['norm_2'])/len(grads[i]['norm_2'])
        grad_norm_1_normalized = sum(grads[i]['normalized_norm_1'])/len(grads[i]['normalized_norm_1'])
        grad_norm_2_normalized = sum(grads[i]['normalized_norm_2'])/len(grads[i]['normalized_norm_2'])

        wandb.log({f"train loss_{i}": train_loss, f"val loss_{i}": val_loss, f'train accuracy_{i}': train_acc, f'val accuracy_{i}': val_acc, f'grad_norm_1_net_{i}': grad_norm_1, f'grad_norm_2_net_{i}': grad_norm_2, f'grad_norm_1_normalized_net_{i}': grad_norm_1_normalized, f'grad_norm_2_normalized_net_{i}': grad_norm_2_normalized, f'epoch':epoch})
    return student, train_loss, val_loss, val_acc

### NEW FUNCTION
def get_samples_certainties(probs, labels):
    confidence = probs.max(dim=1)[0]
    correctness = probs.argmax(dim=1) == labels
    samples_certainties = torch.stack([confidence, correctness.float()], dim=1)
    return samples_certainties

def _log_uncertainty(log_title, samples_certainties, epoch, i):
    with torch.no_grad():
        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        sorted_samples_certainties = samples_certainties[indices_sorting_by_confidence]
        gamma_correlation = uncertainty_metrics.gamma_correlation(sorted_samples_certainties, sort=False)
        wandb.log({f'confidence_statistics/confidence_mean_{log_title}_{i}':uncertainty_metrics.confidence_mean(sorted_samples_certainties),
                f'confidence_statistics/confidence_median_{log_title}_{i}':uncertainty_metrics.confidence_median(sorted_samples_certainties),
                f'confidence_statistics/confidence_gini_{log_title}_{i}':uncertainty_metrics.gini(sorted_samples_certainties),
                f'ranking/auroc_{log_title}_{i}':gamma_correlation['AUROC'], 'epoch':epoch})
        ece = uncertainty_metrics.ECE_calc(sorted_samples_certainties)
        wandb.log({f'ece/ece_{log_title}_{i}':
                 ece[0],'epoch':epoch})
        wandb.log({f'ece/mce_{log_title}_{i}':
                 ece[1], 'epoch':epoch})
