from comet_ml import Experiment
from image_classification.utils import uncertainty_metrics
from tqdm import tqdm

from fastai.vision import *

from image_classification.utils.utils import *
from image_classification.utils import *
import wandb

from datetime import date, datetime

def train(student, teachers_list, data, sf_teacher, sf_student, loss_function, loss_function2, optimizer, hyper_params, epoch, savename, best_val_acc, expt=None):
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    today = date.today().strftime("%d/%m")
    run = wandb.init(
    project="our-awesome-project",
    entity = "ido-shani-proj" ,
    name= f"{hyper_params['experiment']}-{hyper_params['model']}-{hyper_params['num_epochs']} epochs-{today}-{current_time}",
    config=hyper_params)
    config = wandb.config
    if hyper_params['teacher_training']:
        artifact = wandb.Artifact(name=hyper_params['model'], type='model')

    print(hyper_params)
    if type(data) is dict:
        loop = tqdm(data['train_dl'])
    else:
        loop = tqdm(data.train_dl)
    max_val_acc = best_val_acc
    gpu = hyper_params['gpu']
    student.train()
    student = student.to(gpu)
    if teachers_list is not None:
        for teacher in teachers_list:
            teacher.eval()
            teacher = teacher.to(gpu)
    train_loss_list = []
    student_pred_train_list = []
    grad_list = []
    log_teacher_metrics = False
    if teachers_list and len(teachers_list)==1:
        teacher_pred_train_list = []
        log_teacher_metrics = True
    labels_train_list = []
    for images, labels in loop:
        if gpu != 'cpu':
            images = torch.autograd.Variable(images).to(gpu).float()
            labels = torch.autograd.Variable(labels).to(gpu)
        else:
            images = torch.autograd.Variable(images).float()
            labels = torch.autograd.Variable(labels)
        
        student_logits = student(images)
        student_pred_train_list.append(F.softmax(student_logits, dim = 1))
        labels_train_list.append(labels)

        if teachers_list is not None:
            teachers_logits_list = []
            for teacher in teachers_list:
                teacher_logits = teacher(images)
                teachers_logits_list.append(teacher_logits)
                if log_teacher_metrics:
                    teacher_pred_train_list.append(F.softmax(teacher_logits, dim = 1))
            all_teachers_logits = torch.stack(teachers_logits_list, dim=0)
            teacher_mean_logits = torch.mean(all_teachers_logits,dim=0)

        # classifier training
        if teachers_list is None:
            loss = loss_function(student_logits, labels)

        elif expt == 'hinton-kd':
            TEMP = hyper_params['temperature']
            ALPHA = hyper_params['alpha']
            
            teacher_soft_targets = F.softmax(teacher_mean_logits/TEMP,dim=1)
            distillation_loss = loss_function(F.log_softmax(student_logits/TEMP,dim=1), teacher_soft_targets)*(1-ALPHA)*TEMP*TEMP 
            student_loss = loss_function2(F.softmax(student_logits,dim=1),labels)*(ALPHA)
            loss = distillation_loss + student_loss
        train_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate grad size
        total_norm = 0
        count_params = 0
        for p in student.parameters():
            count_params += 1
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item()
        grad_list.append(total_norm/count_params)

        loop.set_description('Epoch {}/{}'.format(epoch + 1, hyper_params['num_epochs']))
        loop.set_postfix(loss=loss.item())
    
    all_student_pred_train = torch.cat(student_pred_train_list)
    all_labels_train = torch.cat(labels_train_list)
    if log_teacher_metrics:
        all_teacher_pred_train = torch.cat(teacher_pred_train_list)
        samples_certainties = get_samples_certainties(all_teacher_pred_train, all_labels_train)
        _log_uncertainty("train_teacher", samples_certainties, epoch)
    samples_certainties = get_samples_certainties(all_student_pred_train, all_labels_train)
    _log_uncertainty("train", samples_certainties, epoch)
    
    _, student_train_pred_final = torch.max(all_student_pred_train, 1)
    if log_teacher_metrics:
        _, teacher_train_pred_final = torch.max(all_teacher_pred_train, 1)
    student_correct = (student_train_pred_final==all_labels_train).sum().item()
    if log_teacher_metrics:
        teacher_correct = (teacher_train_pred_final==all_labels_train).sum().item()
    train_acc = student_correct / all_labels_train.size(0)
    if log_teacher_metrics:
        teacher_train_acc = teacher_correct / all_labels_train.size(0)
    
    train_loss = (sum(train_loss_list) / len(train_loss_list))

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
        for _, (images, labels) in enumerate(tqdm(valid_loop)):
            if gpu != 'cpu':
                images = torch.autograd.Variable(images).to(gpu).float()
                labels = torch.autograd.Variable(labels).to(gpu)
            else:
                images = torch.autograd.Variable(images).float()
                labels = torch.autograd.Variable(labels)

            student_logits = student(images)
            y_pred_val_list.append(F.softmax(student_logits, dim = 1))
            labels_val_list.append(labels)

            if log_teacher_metrics:
                teacher_soft_targets = teachers_list[0](images)
                ##TODO: delete later
                teacher_pred_val_list.append(F.softmax(teacher_soft_targets, dim = 1))
                ##END of delete
            # classifier training
            if teachers_list is None:
                loss = loss_function(student_logits, labels)
                student_logits = F.log_softmax(student_logits, dim = 1)

                _, pred_ind = torch.max(student_logits, 1)

                total += labels.size(0)
                correct += (pred_ind == labels).sum().item()                
            elif expt == 'hinton-kd':
                ALPHA = hyper_params['alpha']

                student_logits = F.log_softmax(student_logits, dim = 1)
                _, pred_ind = torch.max(student_logits, 1)

                total += labels.size(0)
                correct += (pred_ind == labels).sum().item()
                

                loss = loss_function2(student_logits,labels)

            val_loss_list.append(loss.item())
    
    all_y_pred_val = torch.cat(y_pred_val_list)
    _, all_pred_idx = torch.max(all_y_pred_val,1)
    all_labels_val = torch.cat(labels_val_list)
    correct = (all_pred_idx==all_labels_val).sum().item()
    ##TODO: delete later, only to test accuracy of teacher ResNet34 of paper!!
    if log_teacher_metrics:
        all_teacher_pred_val = torch.cat(teacher_pred_val_list)
        _, teacher_val_pred_final = torch.max(all_teacher_pred_val, 1)
        teacher_correct_val = (teacher_val_pred_final==all_labels_val).sum().item()
        teacher_val_acc = teacher_correct_val / all_labels_val.size(0)

    if log_teacher_metrics:
        samples_certainties = get_samples_certainties(all_teacher_pred_val, all_labels_val)
        _log_uncertainty("val_teacher", samples_certainties, epoch)
    samples_certainties = get_samples_certainties(all_y_pred_val, all_labels_val)
    _log_uncertainty("val", samples_certainties, epoch)

    val_loss = (sum(val_loss_list) / len(val_loss_list))
    if total > 0:
        val_acc = correct / total
    else:
        val_acc = None

    # save trained teacher model
    if teachers_list is None:
        if (val_acc * 100) > max_val_acc :
            print(f'higher valid acc obtained: {val_acc * 100}')
            max_val_acc = val_acc * 100
            torch.save(student.state_dict(), savename)
            ## wandb save model
        if epoch == hyper_params['num_epochs'] -1 and hyper_params['teacher_training']:
            artifact.add_file(savename)
            run.log_artifact(artifact)
    # stage training
    elif loss_function2 is None:
        if val_loss < max_val_acc :
            print(f'lower valid loss obtained: {val_loss}')
            max_val_acc = val_loss
            torch.save(student.state_dict(), savename)
    # simultaneous training or attention kd
    else:
        if (val_acc * 100) > max_val_acc :
            print(f'higher valid acc obtained: {val_acc * 100}')
            max_val_acc = val_acc * 100
            torch.save(student.state_dict(), savename)

    wandb.log({"train loss": train_loss, "val loss": val_loss, 'train accuracy': train_acc, 'val accuracy': val_acc, 'grad norm2': sum(grad_list)/len(grad_list), 'epoch':epoch})
    if log_teacher_metrics:
        wandb.log({'teacher_train accuracy': teacher_train_acc, 'teacher_val accuracy': teacher_val_acc, 'epoch':epoch})
    return student, train_loss, val_loss, val_acc, max_val_acc

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
