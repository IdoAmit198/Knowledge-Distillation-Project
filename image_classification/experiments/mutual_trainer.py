from comet_ml import Experiment
from image_classification.utils import uncertainty_metrics
from tqdm import tqdm

from fastai.vision import *

from image_classification.utils.utils import *
from image_classification.utils import *
import wandb

from datetime import date, datetime

def mutual_train(mutual_nets, data, loss_function, loss_function2, optimizers, hyper_params, epoch, savename, best_val_acc, expt=None):
    # print(f'is two nets the same, is mutual_nets[0] is mutual_nets[1]: {mutual_nets[0] is mutual_nets[1]}')
    # print(f'mutual_nets[0] id is: {id(mutual_nets[0])}')
    # print(f'mutual_nets[1] id is: {id(mutual_nets[1])}')
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    today = date.today().strftime("%d/%m")
    run = wandb.init(
    project="our-awesome-project",
    entity = "ido-shani-proj" ,
    # group=f"{hyper_params.experiment}",
    name= f"{hyper_params['experiment']}-{hyper_params['model']}-{hyper_params['num_epochs']} epochs-{today}-{current_time}",
    config=hyper_params)
    config = wandb.config

    print(hyper_params)
    if type(data) is dict:
        loop = tqdm(data['train_dl'])
    else:
        loop = tqdm(data.train_dl)
    max_val_acc = best_val_acc
    gpu = hyper_params['gpu']
    mutual_networks_num = len(mutual_nets)

    train_loss_list = [[] for x in range(mutual_networks_num)]
    student_pred_train_list = [[] for x in range(mutual_networks_num)]
    labels_train_list = [[] for x in range(mutual_networks_num)]
    print(f"empty train_loss_list: {train_loss_list}")
    print(f"empty student_pred_train_list: {student_pred_train_list}")
    print(f"empty labels_train_list: {labels_train_list}")
    for images, labels in loop:
        for i in range(mutual_networks_num):
            # if idx == 3:
            #     break
            teachers_list = [mutual_nets[j] for j in range(mutual_networks_num) if j != i]
            print(f"teachers_list Len= {len(teachers_list)}")
            if len(teachers_list) != mutual_networks_num - 1:
                print(f"number of teachers for student {i} is: {len(teachers_list)}")
            # student = mutual_nets[i]
            student = mutual_nets[i]
            
            print(f'student id is: {id(student)}')
            for teacher in teachers_list:
                print(f'teacher id is: {id(teacher)}')

            for param in student.parameters():
                param.requires_grad = True
            student.train()
            student = student.to(gpu)
            for teacher in teachers_list:
                for param in teacher.parameters():
                    param.requires_grad = False
                teacher.eval()
                teacher = teacher.to(gpu)

            if gpu != 'cpu':
                images = torch.autograd.Variable(images).to(gpu).float()
                labels = torch.autograd.Variable(labels).to(gpu)
            else:
                images = torch.autograd.Variable(images).float()
                labels = torch.autograd.Variable(labels)
            
            student_logits = student(images)
            student_pred_train_list[i].append(F.softmax(student_logits, dim = 1))
            labels_train_list[i].append(labels)

            teachers_logits_list = []
            for teacher in teachers_list:
                teacher_logits = teacher(images)
                teachers_logits_list.append(teacher_logits)
            
            all_teachers_logits = torch.stack(teachers_logits_list, dim=0)
            teacher_mean_logits = torch.mean(all_teachers_logits,dim=0)

            # stage training (and assuming sf_teacher and sf_student are given)

            TEMP = hyper_params['temperature']
            ALPHA = hyper_params['alpha']

            teacher_soft_targets = F.softmax(teacher_mean_logits/TEMP,dim=1)
            distillation_loss = loss_function(F.log_softmax(student_logits/TEMP,dim=1), teacher_soft_targets)*(1-ALPHA)*TEMP*TEMP 
            student_loss = loss_function2(F.softmax(student_logits,dim=1),labels)*(ALPHA)
            loss = distillation_loss + student_loss

            train_loss_list[i].append(loss.item())

            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()
            # for i, param in enumerate(student.parameters()):
            #     printf(f'param{i}: {param}')
                # if (param_list[i]==param.items()):
                #     print(f'error, equal params: {i}')

            loop.set_description('Epoch {}/{} student_{}'.format(epoch + 1, hyper_params['num_epochs'], i))
            loop.set_postfix(loss=loss.item())
            
            mutual_nets[i] = student
    
    for net in mutual_nets:
        net.eval()

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for i in range(mutual_networks_num):
        all_student_pred_train = torch.cat(student_pred_train_list[i])
        all_labels_train = torch.cat(labels_train_list[i])
        samples_certainties = get_samples_certainties(all_student_pred_train, all_labels_train)
        _log_uncertainty(f"train_{i}", samples_certainties, epoch)
        
        _, student_train_pred_final = torch.max(all_student_pred_train, 1)
        student_correct = (student_train_pred_final==all_labels_train).sum().item()
        train_acc.append(student_correct / all_labels_train.size(0))
        
        train_loss.append(sum(train_loss_list[i]) / len(train_loss_list[i]))

        mutual_nets[i].eval()
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
                
                student_logits = mutual_nets[i](images)
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

        samples_certainties = get_samples_certainties(all_y_pred_val, all_labels_val)
        _log_uncertainty("val", samples_certainties, epoch)

        val_loss.append(sum(val_loss_list) / len(val_loss_list))
        if total > 0:
            val_acc.append(correct / total)
            print(f"## val_acc = val_acc")
        else:
            val_acc.append(None)

        if (val_acc[i] * 100) > max_val_acc[i] :
            print(f'higher valid acc obtained: {val_acc[i] * 100}')
            max_val_acc[i] = val_acc[i] * 100

        print(f'train_loss: {train_loss}')
        print(f'val_loss: {val_loss}')
        print(f'train_acc: {train_acc}')
        print(f'val_acc: {val_acc}')
        print(f'correct: {correct}')
        print(f'total: {total}')
        print(f'epoch: {epoch}')
        wandb.log({f"train loss_{i}": train_loss[i], f"val loss_{i}": val_loss[i], f'train accuracy_{i}': train_acc[i], f'val accuracy_{i}': val_acc[i], f'epoch_{i}':epoch})
    return mutual_nets, train_loss, val_loss, val_acc, max_val_acc

### NEW FUNCTION
def get_samples_certainties(probs, labels):
    # probs = F.softmax(preds, dim=1)
    confidence = probs.max(dim=1)[0]
    correctness = probs.argmax(dim=1) == labels
    samples_certainties = torch.stack([confidence, correctness.float()], dim=1)
    return samples_certainties

def _log_uncertainty(log_title, samples_certainties, epoch):
    with torch.no_grad():
        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        sorted_samples_certainties = samples_certainties[indices_sorting_by_confidence]
        gamma_correlation = uncertainty_metrics.gamma_correlation(sorted_samples_certainties, sort=False)
        wandb.log({f'confidence_statistics/confidence_mean_{log_title}':uncertainty_metrics.confidence_mean(sorted_samples_certainties),
                f'confidence_statistics/confidence_median_{log_title}':uncertainty_metrics.confidence_median(sorted_samples_certainties),
                f'confidence_statistics/confidence_gini_{log_title}':uncertainty_metrics.gini(sorted_samples_certainties),
                f'ranking/auroc_{log_title}':gamma_correlation['AUROC'], 'epoch':epoch})
        ece = uncertainty_metrics.ECE_calc(sorted_samples_certainties)
        wandb.log({f'ece/ece_{log_title}':
                 ece[0],'epoch':epoch})
        wandb.log({f'ece/mce_{log_title}':
                 ece[1], 'epoch':epoch})
