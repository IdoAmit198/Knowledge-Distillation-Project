from comet_ml import Experiment
from image_classification.utils import uncertainty_metrics
from tqdm import tqdm

from fastai.vision import *

from image_classification.utils.utils import *
from image_classification.utils import *
import wandb

from datetime import date, datetime

def train(student_embedding, teachers_list, fc_heads, data, loss_function, loss_function2, optimizer, hyper_params, epoch, savename, best_val_acc, expt=None):
    """
    trainer function for multi-head student against multiple teachers.
    In this regime, the student has a different FC head for ground-truth and for teachers.
    Can handle two cases:
    1. multiple teachers aggregated to single logits.
    2. multiple teachers not aggregated and treated w.r.t each one seperately.
    -----------
    parameters:
    student_embedding - The student embedding without a FC heads. Used by all heads.
    teachers_list - A list of teachers to learn from.
    fc_heads -  dict of FC heads. the head to compare against ground truth is 'gt_head'.
                In case of aggregate_teachers, the head to compare against average of teachers is 'mean_teachers_head'.
                In case of not aggregate_teachers, the heads to compare against teachers are teacher0_head, teacher1_head and so on.
    data - the dataloaders, as always.
    loss_function - The loss function to use for distillation, against teachers logits. Usually KLDivLoss.
    loss_function2 - The loss function to use for ground truth. Usually CrossEntropyLoss.
    optimizer - Optimizer to use. It is equipped with parameters of student_embedding as well as all fc_heads.
    """
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
    if hyper_params['teacher_training']:
        artifact = wandb.Artifact(name=hyper_params['model'], type='model')

    print(hyper_params)
    if type(data) is dict:
        loop = tqdm(data['train_dl'])
    else:
        loop = tqdm(data.train_dl)
    max_val_acc = best_val_acc
    gpu = hyper_params['gpu']
    student_embedding.train()
    for head in fc_heads.values():
        head.train()
        head = head.to(gpu)
    student_embedding = student_embedding.to(gpu)
    if teachers_list is not None:
        #Consider move the teachers in and out manually.
        for teacher in teachers_list:
            teacher.eval()
            teacher = teacher.to(gpu)
    train_loss_list = []
    # heads_pred_train_list = []
    all_heads_pred_train_dict = {}
    mean_heads_pred_train_list = []
    gt_head_pred_train_list = []
    log_teacher_metrics = False
    if teachers_list and len(teachers_list)>=1:
        teachers_pred_train_list = []
        log_teacher_metrics = True
    labels_train_list = []
    for images, labels in loop:
        # if gpu != 'cpu':
        images = torch.autograd.Variable(images).to(gpu).float()
        labels = torch.autograd.Variable(labels).to(gpu)
        # else:
        #     images = torch.autograd.Variable(images).float()
        #     labels = torch.autograd.Variable(labels)
        
        embed_logits = student_embedding(images)
        gt_head_logits = fc_heads['gt_head'](embed_logits)
        # We collect all into a big list of all batches for later logs
        gt_head_pred_train_list.append(F.softmax(gt_head_logits, dim = 1))
        if hyper_params['aggregate_teachers']:
            mean_teachers_head_logits = fc_heads['mean_teachers_head'](embed_logits)
            # We collect all into a big list of all batches for later logs 
            mean_heads_pred_train_list.append(F.softmax(mean_teachers_head_logits, dim = 1))
        else:
            #get a list of all other head_logits
            all_teachers_heads_logits = [(head_name,head(embed_logits)) for (head_name,head) in fc_heads.items()\
                                        if head_name!='gt_head']
            all_teachers_heads_probs = [(head_name,F.softmax((logit), dim = 1)) for (head_name,logit) in all_teachers_heads_logits]
            all_teachers_heads_logits = [logit for (_, logit) in all_teachers_heads_logits]
            # print(f"\n len(all_teachers_heads_probs)={len(all_teachers_heads_probs)}")
            # We collect all into a big list of all batches for later logs
            for head_logits in all_teachers_heads_probs:
                try:
                    all_heads_pred_train_dict[head_logits[0]].append(head_logits[1])
                except KeyError:
                    all_heads_pred_train_dict[head_logits[0]] = [head_logits[1]]
            # print(f"\n all_heads_pred_train_dict['resnet34_1_head'].shape \n = {all_heads_pred_train_dict['resnet34_1_head'].shape} \n")
            # assert 1==0, "Done!"
        
        labels_train_list.append(labels)

        if teachers_list:
            teachers_logits_list = []
            for teacher in teachers_list:
                teacher_logits = teacher(images)
                teachers_logits_list.append(teacher_logits)
                if log_teacher_metrics:
                    teachers_pred_train_list.append(F.softmax(teacher_logits, dim = 1))
            all_teachers_logits = torch.stack(teachers_logits_list, dim=0)
            if hyper_params['aggregate_teachers']:
                teacher_mean_logits = torch.mean(all_teachers_logits,dim=0)

        if expt == 'student_MH_kd':
            TEMP = hyper_params['temperature']
            ALPHA = hyper_params['alpha']
            distillation_losses_list = []
            if hyper_params['aggregate_teachers']:
                teacher_soft_targets = F.softmax(teacher_mean_logits/TEMP,dim=1)
                distillation_losses_list.append(loss_function(F.log_softmax(mean_teachers_head_logits/TEMP,dim=1), teacher_soft_targets)\
                                                *(1-ALPHA)*TEMP*TEMP)
            else:
                teachers_soft_targets = [F.softmax(teacher_logits/TEMP,dim=1) for teacher_logits in all_teachers_logits]
                for (head_logits, teacher_soft_targets) in zip(all_teachers_heads_logits, teachers_soft_targets):
                    distillation_losses_list.append(loss_function(F.log_softmax(head_logits/TEMP,dim=1), teacher_soft_targets)*(1-ALPHA)*TEMP*TEMP)
                distillation_losses_list = torch.tensor(distillation_losses_list, device=gpu)
            ground_truth_loss = loss_function2(F.softmax(gt_head_logits,dim=1),labels)*(ALPHA)
            ground_truth_loss = ground_truth_loss.to(gpu)
            loss = torch.mean(distillation_losses_list) + ground_truth_loss
            loss = loss.to(gpu)
        else:
            raise Exception(f"ERROR! expt should be \'student_MH_kd\', but got {expt}.")

        train_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description('Epoch {}/{}'.format(epoch + 1, hyper_params['num_epochs']))
        loop.set_postfix(loss=loss.item())
    
    labels_train_list = torch.cat(labels_train_list)
    gt_head_pred_train = torch.cat(gt_head_pred_train_list)
    gt_head_samples_certainties = get_samples_certainties(gt_head_pred_train, labels_train_list)
    _log_uncertainty("train ground truth head", gt_head_samples_certainties, epoch)
    _, gt_head_train_pred_final = torch.max(gt_head_pred_train, 1)
    gt_head_correct = (gt_head_train_pred_final==labels_train_list).sum().item()
    gt_head_train_acc = gt_head_correct / labels_train_list.size(0)
    
    if hyper_params['aggregate_teachers']:
        mean_heads_pred_train = torch.cat(mean_heads_pred_train_list)
        mean_heads_samples_certainties = get_samples_certainties(mean_heads_pred_train, labels_train_list)
        _log_uncertainty("train teachers mean head", mean_heads_samples_certainties, epoch)
        _, mean_head_train_pred_final = torch.max(mean_heads_pred_train, 1)
        mean_head_correct = (mean_head_train_pred_final==labels_train_list).sum().item()
        mean_head_train_acc = mean_head_correct / labels_train_list.size(0)
    else:
        ## Below logging certainties of different heads.
        ## The on below is a bit probelamtic in case of multiple teachers.
        ## Need to figure it out.
        for name_head,head_pred_train_tensors in all_heads_pred_train_dict.items():
            all_heads_pred_train_dict[name_head] = torch.cat(head_pred_train_tensors)
        all_heads_samples_certainties_dict = {name_head:get_samples_certainties(head_pred_train, labels_train_list) for \
                                            name_head,head_pred_train in all_heads_pred_train_dict.items()}
        for name_head,head_samples_certainties in all_heads_samples_certainties_dict.items():
            _log_uncertainty(f"train {name_head}", head_samples_certainties, epoch)
        all_heads_pred_train_dict_final = {name_head:(torch.max(head_pred_train_tensors, 1))[1] for \
                                        name_head,head_pred_train_tensors in all_heads_pred_train_dict.items()}
        # print(f"\n all_heads_pred_train_dict_final['resnet34_1_head'].shape={all_heads_pred_train_dict_final['resnet34_1_head'].shape} \n")
        all_heads_pred_train_correct = {f"{name_head}_correct":(head_pred_train_final==labels_train_list).sum().item() for \
                                        name_head,head_pred_train_final in all_heads_pred_train_dict_final.items()}
        all_heads_pred_train_acc = {f"{name_head}_acc":head_pred_train_correct/labels_train_list.size(0) for \
                                        name_head,head_pred_train_correct in all_heads_pred_train_correct.items()}
        for name,acc in all_heads_pred_train_acc.items():
            print(f"name is {acc}")    
    ## Below logging teacher metrics: now logging in though there is one teacher.
    ## Need to adjust for multiple teachers.
    # if log_teacher_metrics:
    #     all_teacher_pred_train = torch.cat(teachers_pred_train_list)
    #     samples_certainties = get_samples_certainties(all_teacher_pred_train, labels_train_list)
    #     _log_uncertainty("train_teacher", samples_certainties, epoch)  
    
    ## Logging teachers below
    #TODO: fix logging teachers here and above.
    # if log_teacher_metrics:
    #     _, teacher_train_pred_final = torch.max(all_teacher_pred_train, 1)
    ## Logging teachers below
    #TODO: fix logging teachers here and above.
    # if log_teacher_metrics:
    #     teacher_correct = (teacher_train_pred_final==labels_train_list).sum().item()
    ## Logging teachers below
    #TODO: fix logging teachers here and above.
    # if log_teacher_metrics:
    #     teacher_train_acc = teacher_correct / labels_train_list.size(0)
    
    train_loss = (sum(train_loss_list) / len(train_loss_list))

    # wandb.log({"train loss": train_loss, 'ground truth head train accuracy': gt_head_train_acc,\
    #             'epoch':epoch})
    # return student_embedding, fc_heads, train_loss, train_loss, max_val_acc, max_val_acc

    val_loss_list = list()
    with torch.no_grad():
        student_embedding.eval()
        for head in fc_heads.values():
            head.eval()
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
        if epoch == hyper_params['num_epochs'] -1:
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

    wandb.log({"train loss": train_loss, "val loss": val_loss, 'train accuracy': train_acc, 'val accuracy': val_acc, 'epoch':epoch})
    if log_teacher_metrics:
        wandb.log({'teacher_train accuracy': teacher_train_acc, 'teacher_val accuracy': teacher_val_acc, 'epoch':epoch})
    return student_embedding, fc_heads, train_loss, val_loss, val_acc, max_val_acc

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
