from comet_ml import Experiment
from image_classification.utils import uncertainty_metrics
from tqdm import tqdm

from fastai.vision import *

from image_classification.utils.utils import *
from image_classification.utils import *
import wandb

from datetime import date, datetime


def train(student, teacher, data, sf_teacher, sf_student, loss_function, loss_function2, optimizer, hyper_params, epoch, savename, best_val_acc, expt=None):
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    today = date.today().strftime("%d/%m")
    wandb.init(
    project="our-awesome-project",
    # group=f"{hyper_params.experiment}",
    name= f"{hyper_params['experiment']}-{hyper_params['model']}-{hyper_params['num_epochs']} epochs-{today}-{current_time}",
    config=hyper_params)
    config = wandb.config
    
    loop = tqdm(data.train_dl)
    max_val_acc = best_val_acc
    gpu = hyper_params['gpu']
    student.train()
    student = student.to(gpu)
    if teacher is not None:
        teacher.eval()
        teacher = teacher.to(gpu)
    trn = list()
    y_pred_train_list = []
    labels_train_list = []
    for images, labels in loop:
        # if idx == 3:
        #     break

        if gpu != 'cpu':
            images = torch.autograd.Variable(images).to(gpu).float()
            labels = torch.autograd.Variable(labels).to(gpu)
        else:
            images = torch.autograd.Variable(images).float()
            labels = torch.autograd.Variable(labels)
        
        ## y_pred is logits. duh. 
        y_pred = student(images)
        # print(y_pred.shape)
        y_pred_train_list.append(y_pred)
        labels_train_list.append(labels)
        ## CODE WE CHANGED
        # print(f"y_pred = {y_pred}")
        # print(f"y_pred shape is {y_pred.shape}")
        # print(f"y_pred sum is {torch.sum(y_pred)}")

        
        ## ENF OF CODE WE CHANGED

        if teacher is not None:
            soft_targets = teacher(images)

        # classifier training
        if teacher is None:
            loss = loss_function(y_pred, labels)
        # stage training (and assuming sf_teacher and sf_student are given)

        elif expt == 'hinton-kd':
            TEMP = hyper_params['temperature']
            ALPHA = hyper_params['alpha']
            
            soft_targets = F.softmax(soft_targets/TEMP,dim=1)
            loss = loss_function(F.log_softmax(y_pred/TEMP,dim=1),soft_targets)*(1-ALPHA)*TEMP*TEMP
            loss += loss_function2(F.softmax(y_pred,dim=1),labels)*(ALPHA)


        elif loss_function2 is None:
            if expt == 'fsp-kd':
                loss = 0
                # 4 intermediate feature maps and taken 2 at a time (thus 3)
                for k in range(3):
                    loss += loss_function(fsp_matrix(sf_teacher[k].features, sf_teacher[k + 1].features),
                                          fsp_matrix(sf_student[k].features, sf_student[k + 1].features))
                loss /= 3
            else:
                loss = loss_function(sf_student[hyper_params['stage']].features, sf_teacher[hyper_params['stage']].features)
        # attention transfer KD
        elif expt == 'attention-kd':
            loss = loss_function(y_pred, labels)
            for k in range(4):
                loss += loss_function2(at(sf_student[k].features), at(sf_teacher[k].features))
            loss /= 5
        # 2 loss functions and student and teacher are given -> simultaneous training
        else:
            loss = loss_function(y_pred, labels)
            for k in range(5):
                loss += loss_function2(sf_student[k].features, sf_teacher[k].features)
            # normalizing factor (doesn't affect optimization theoretically)
            loss /= 6

        trn.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description('Epoch {}/{}'.format(epoch + 1, hyper_params['num_epochs']))
        loop.set_postfix(loss=loss.item())
    
    all_y_pred_train = torch.cat(y_pred_train_list)
    all_labels_train = torch.cat(labels_train_list)
    # print(f'all_y_pred shape: {all_y_pred.shape}')
    # print(f'all_labels shape: {all_labels.shape}')
    samples_certainties = get_samples_certainties(all_y_pred_train, all_labels_train)
    _log_uncertainty("train", samples_certainties)
    
    train_loss = (sum(trn) / len(trn))

    student.eval()
    val = list()
    with torch.no_grad():
        correct = 0
        total = 0

        y_pred_val_list = []
        labels_val_list = []
        for _, (images, labels) in enumerate(data.valid_dl):
            if gpu != 'cpu':
                images = torch.autograd.Variable(images).to(gpu).float()
                labels = torch.autograd.Variable(labels).to(gpu)
            else:
                images = torch.autograd.Variable(images).float()
                labels = torch.autograd.Variable(labels)

            y_pred = student(images)
            
            y_pred_val_list.append(y_pred)
            labels_val_list.append(labels)

            if teacher is not None:
              soft_targets = teacher(images)

            # classifier training
            if teacher is None:
                loss = loss_function(y_pred, labels)
                y_pred = F.log_softmax(y_pred, dim = 1)

                _, pred_ind = torch.max(y_pred, 1)

                total += labels.size(0)
                correct += (pred_ind == labels).sum().item()
            
            elif expt == 'hinton-kd':
                ALPHA = hyper_params['alpha']

                y_pred = F.log_softmax(y_pred, dim = 1)
                _, pred_ind = torch.max(y_pred, 1)

                total += labels.size(0)
                correct += (pred_ind == labels).sum().item()

                loss = loss_function2(y_pred,labels)


            # stage training
            elif loss_function2 is None:
                if expt == 'fsp-kd':
                    loss = 0
                    # 4 intermediate feature maps and taken 2 at a time (thus 3)
                    for k in range(3):
                        loss += loss_function(fsp_matrix(sf_teacher[k].features, sf_teacher[k + 1].features),
                                              fsp_matrix(sf_student[k].features, sf_student[k + 1].features))
                    loss /= 3
                else:
                    loss = loss_function(sf_student[hyper_params['stage']].features, sf_teacher[hyper_params['stage']].features)
            # simultaneous training or attention KD
            else:
                loss = loss_function(y_pred, labels)
                y_pred = F.log_softmax(y_pred, dim = 1)

                _, pred_ind = torch.max(y_pred, 1)

                total += labels.size(0)
                correct += (pred_ind == labels).sum().item()

            val.append(loss.item())

    all_y_pred_val = torch.cat(y_pred_val_list)
    all_labels_val = torch.cat(labels_val_list)
    samples_certainties = get_samples_certainties(all_y_pred_val, all_labels_val)
    _log_uncertainty("val", samples_certainties)

    val_loss = (sum(val) / len(val))
    if total > 0:
        val_acc = correct / total
    else:
        val_acc = None

    # classifier training
    if teacher is None:
        if (val_acc * 100) > max_val_acc :
            print(f'higher valid acc obtained: {val_acc * 100}')
            max_val_acc = val_acc * 100
            torch.save(student.state_dict(), savename)
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

    wandb.log({"train loss": train_loss, "val loss": val_loss, 'val accuracy': val_acc, 'epoch':epoch})
    return student, train_loss, val_loss, val_acc, max_val_acc

### NEW FUNCTION
def get_samples_certainties(preds, labels):
        probs = F.softmax(preds, dim=1)
        confidence = probs.max(dim=1)[0]
        correctness = probs.argmax(dim=1) == labels
        samples_certainties = torch.stack([confidence, correctness.float()], dim=1)
        return samples_certainties

def _log_uncertainty(log_title, samples_certainties):
    with torch.no_grad():
        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        sorted_samples_certainties = samples_certainties[indices_sorting_by_confidence]
        gamma_correlation = uncertainty_metrics.gamma_correlation(sorted_samples_certainties, sort=False)
        wandb.log({f'confidence_statistics/confidence_mean_{log_title}':uncertainty_metrics.confidence_mean(sorted_samples_certainties),
                f'confidence_statistics/confidence_median_{log_title}':uncertainty_metrics.confidence_median(sorted_samples_certainties),
                f'confidence_statistics/confidence_gini_{log_title}':uncertainty_metrics.gini(sorted_samples_certainties),
                f'ranking/auroc_{log_title}':gamma_correlation['AUROC']})
        ##TODO: Veirfy to get it work later
        ece = uncertainty_metrics.ECE_calc(sorted_samples_certainties)
        wandb.log({f'ece/ece_{log_title}':
                 ece[0]})
        wandb.log({f'ece/mce_{log_title}':
                 ece[1]})
