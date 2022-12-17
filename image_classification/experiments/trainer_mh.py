from comet_ml import Experiment
from image_classification.utils import uncertainty_metrics
from tqdm import tqdm

from fastai.vision import *

from image_classification.utils.utils import *
from image_classification.utils import *
import wandb

from datetime import date, datetime

def train(student_embedding, teachers_dict, fc_heads, data, loss_function, loss_function2, \
            optimizer, hyper_params, epoch, savename, best_val_acc, batches_train_logits_teachers, \
            batches_val_logits_teachers, all_teachers_samples_certainties_dict, teachers_train_acc_dict, \
            teachers_val_acc_dict, mean_teachers_train_acc, expt=None):
    """
    trainer function for multi-head student against multiple teachers.
    In this regime, the student has a different FC head for ground-truth and for teachers.
    Can handle two cases:
    1. multiple teachers aggregated to single logits.
    2. multiple teachers not aggregated and treated w.r.t each one seperately.
    -----------
    parameters:
    student_embedding - The student embedding without a FC heads. Used by all heads.
    teachers_dict - A dict of (key,value) of teacher_name:teacher_model to learn from.
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
    if teachers_dict is not None:
        #Consider move the teachers in and out manually.
        for _,teacher in teachers_dict.items():
            teacher.eval()
            teacher = teacher.to(gpu)
    train_loss_list = []
    # heads_pred_train_list = []
    all_heads_pred_train_dict = {}
    teachers_logits_dict = {}
    mean_heads_pred_train_list = []
    gt_head_pred_train_list = []
    log_teacher_metrics = False
    teachers_pred_train_dict = {}
    log_teacher_metrics = True
    labels_train_list = []
    for batch_idx,(images, labels) in enumerate(loop):
        if batch_idx==5:
            break
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

        if teachers_dict and epoch==0:
            # teachers_logits_list = []
            for teacher_name,teacher in teachers_dict.items():
                teacher_logits = teacher(images)
                try:
                    teachers_logits_dict[teacher_name].append(teacher_logits)
                except KeyError:
                    teachers_logits_dict[teacher_name] = [teacher_logits]
                try:
                    batches_train_logits_teachers[batch_idx].append(teacher_logits)
                except IndexError:
                    batches_train_logits_teachers.append([teacher_logits])
                if log_teacher_metrics:
                    try:
                        teachers_pred_train_dict[teacher_name].append(F.softmax(teacher_logits, dim = 1))
                    except KeyError:
                        teachers_pred_train_dict[teacher_name] = [F.softmax(teacher_logits, dim = 1)]
            if hyper_params['aggregate_teachers']:
                batches_train_logits_teachers[batch_idx] = torch.mean(torch.stack(batches_train_logits_teachers[batch_idx], dim=0),dim=0)

        if expt == 'student_MH_kd':
            TEMP = hyper_params['temperature']
            ALPHA = hyper_params['alpha']
            distillation_losses_list = []
            if hyper_params['aggregate_teachers']:
                teacher_soft_targets = F.softmax(batches_train_logits_teachers[batch_idx]/TEMP,dim=1)
                distillation_losses_list.append(loss_function(F.log_softmax(mean_teachers_head_logits/TEMP,dim=1), teacher_soft_targets)\
                                                *(1-ALPHA)*TEMP*TEMP)
            else:
                teachers_soft_targets = [F.softmax(teacher_logits/TEMP,dim=1) for teacher_logits in batches_train_logits_teachers[batch_idx]]
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
        # Figured it out and probably works well.
        for name_head,head_pred_train_tensors in all_heads_pred_train_dict.items():
            all_heads_pred_train_dict[name_head] = torch.cat(head_pred_train_tensors)
        all_heads_samples_certainties_dict = {name_head:get_samples_certainties(head_pred_train, labels_train_list) for \
                                            name_head,head_pred_train in all_heads_pred_train_dict.items()}
        for name_head,head_samples_certainties in all_heads_samples_certainties_dict.items():
            _log_uncertainty(f"train_{name_head}", head_samples_certainties, epoch)
        all_heads_pred_train_dict_final = {name_head:(torch.max(head_pred_train_tensors, 1))[1] for \
                                        name_head,head_pred_train_tensors in all_heads_pred_train_dict.items()}
        all_heads_pred_train_correct = {f"{name_head}_correct":(head_pred_train_final==labels_train_list).sum().item() for \
                                        name_head,head_pred_train_final in all_heads_pred_train_dict_final.items()}
        all_heads_pred_train_acc = {f"{name_head}_acc":head_pred_train_correct/labels_train_list.size(0) for \
                                        name_head,head_pred_train_correct in all_heads_pred_train_correct.items()}

    # Below logging teacher metrics: Already checked for multiple heads.
    # Need to check the mean of mean of all teachers.
    if log_teacher_metrics and epoch==0:
        if hyper_params['aggregate_teachers']:
            ## Need to check this one!!
            all_teachers_prob_tensor = torch.stack(batches_train_logits_teachers, dim=0)
            all_teachers_prob_tensor = all_teachers_prob_tensor.reshape(all_teachers_prob_tensor.shape[0]*all_teachers_prob_tensor.shape[1], -1)
            mean_teachers_samples_certainties = get_samples_certainties(all_teachers_prob_tensor, labels_train_list)
            _, mean_teachers_train_pred = torch.max(all_teachers_prob_tensor, 1)
            mean_teachers_train_correct = (mean_teachers_train_pred==labels_train_list).sum().item()
            mean_teachers_train_acc = mean_teachers_train_correct / labels_train_list.size(0)
        else:
            all_teachers_prob_dict = {teacher_name:(torch.stack(teacher_prob_list, dim=0)) \
                                    for teacher_name,teacher_prob_list in teachers_pred_train_dict.items()}
            all_teachers_prob_dict = {teacher_name:(teac_prob_tensor.reshape(teac_prob_tensor.shape[0]*teac_prob_tensor.shape[1], -1)) \
                                    for teacher_name,teac_prob_tensor in all_teachers_prob_dict.items()}
            all_teachers_samples_certainties_dict = {teach_name:get_samples_certainties(teach_prob_train, labels_train_list) for \
                                                teach_name,teach_prob_train in all_teachers_prob_dict.items()}
            for teacher_name, teacher_probs in all_teachers_prob_dict.items():
                _, teacher_train_pred = torch.max(teacher_probs, 1)
                teacher_train_correct = (teacher_train_pred==labels_train_list).sum().item()
                teacher_train_acc = teacher_train_correct / labels_train_list.size(0)
                teachers_train_acc_dict[teacher_name] = teacher_train_acc

    # Below we log cerntainty and compute accuracy for mean_teachers or each teacher,
    # depends whether aggregate teachers or not.
    if hyper_params['aggregate_teachers']:
        _log_uncertainty(f"train_mean_teachers", mean_teachers_samples_certainties, epoch)
    else:
        for teacher_name, teacher_samples_certainties in all_teachers_samples_certainties_dict.items():
            _log_uncertainty(f"train_teacher_{teacher_name}", teacher_samples_certainties, epoch)
    
    train_loss = (sum(train_loss_list) / len(train_loss_list))

    # Start of validation loop
    val_loss_list = list()
    with torch.no_grad():
        student_embedding.eval()
        for head in fc_heads.values():
            head.eval()
        correct = 0
        total = 0

        gt_head_val_logits_list = []
        labels_val_list = []
        mean_heads_prob_val_list = []
        all_heads_prob_val_dict = {}
        teachers_val_logits_dict = {}
        teachers_pred_val_dict = {}
        # teacher_pred_val_list = []
        if type(data) is dict:
            valid_loop = data['valid_dl']
        else:
            valid_loop = data.valid_dl
        for batch_idx, (images, labels) in enumerate(tqdm(valid_loop)):
            images = torch.autograd.Variable(images).to(gpu).float()
            labels = torch.autograd.Variable(labels).to(gpu)

            embed_logits = student_embedding(images)
            gt_head_val_logits_list.append(fc_heads['gt_head'](embed_logits))
            # gt_head_probs_val_list = [F.softmax(gt_head_logit, dim = 1) for gt_head_logit in gt_head_val_logits_list]
            if hyper_params['aggregate_teachers']:
                mean_teachers_val_head_logits = fc_heads['mean_teachers_head'](embed_logits)
                # We collect all into a big list of all batches for later logs 
                mean_heads_prob_val_list.append(F.softmax(mean_teachers_val_head_logits, dim = 1))
            else:
                #get a list of all other head_logits
                all_teachers_heads_logits = [(head_name,head(embed_logits)) for (head_name,head) in fc_heads.items()\
                                            if head_name!='gt_head']
                all_teachers_heads_probs = [(head_name,F.softmax((logit), dim = 1)) for (head_name,logit) in all_teachers_heads_logits]
                all_teachers_heads_logits = [logit for (_, logit) in all_teachers_heads_logits]
                # print(f"\n len(all_teachers_heads_probs)={len(all_teachers_heads_probs)}")
                # We collect all into a big list of all batches for later logs
                for head_prob in all_teachers_heads_probs:
                    try:
                        all_heads_prob_val_dict[head_prob[0]].append(head_prob[1])
                    except KeyError:
                        all_heads_prob_val_dict[head_prob[0]] = [head_prob[1]]

            # y_pred_val_list.append(F.softmax(student_logits, dim = 1))
            labels_val_list.append(labels)
            
            if teachers_dict and epoch==0:
            # teachers_logits_list = []
                for teacher_name,teacher in teachers_dict.items():
                    teacher_val_logits = teacher(images)
                    try:
                        teachers_val_logits_dict[teacher_name].append(teacher_val_logits)
                    except KeyError:
                        teachers_val_logits_dict[teacher_name] = [teacher_val_logits]
                    try:
                        batches_val_logits_teachers[batch_idx].append(teacher_val_logits)
                    except IndexError:
                        batches_val_logits_teachers.append([teacher_val_logits])
                    if log_teacher_metrics:
                        try:
                            teachers_pred_val_dict[teacher_name].append(F.softmax(teacher_val_logits, dim = 1))
                        except KeyError:
                            teachers_pred_val_dict[teacher_name] = [F.softmax(teacher_val_logits, dim = 1)]
                if hyper_params['aggregate_teachers']:
                    batches_val_logits_teachers[batch_idx] = torch.mean(torch.stack(batches_val_logits_teachers[batch_idx], dim=0),dim=0)

            ## Might be not relevant anymore...
            # if log_teacher_metrics:
            #     teacher_soft_targets = teachers_list[0](images)
            #     teacher_pred_val_list.append(F.softmax(teacher_soft_targets, dim = 1))
        
            if expt == 'student_MH_kd':
                ALPHA = hyper_params['alpha']
                # gt_head_val_logits = torch.stack(gt_head_val_logits_list,dim=0)
                # gt_head_val_logits = gt_head_val_logits.reshape(gt_head_val_logits.shape[0]*gt_head_val_logits.shape)
                gt_head_val_soft_logits = F.log_softmax(gt_head_val_logits_list[batch_idx], dim = 1)
                distillation_losses_list = []
                if hyper_params['aggregate_teachers']:
                    teacher_soft_targets = F.softmax(batches_val_logits_teachers[batch_idx],dim=1)
                    distillation_losses_list.append(loss_function(F.log_softmax(mean_teachers_val_head_logits,dim=1), teacher_soft_targets)\
                                                    *(1-ALPHA))
                else:
                    teachers_soft_targets = [F.softmax(teacher_logits,dim=1) for teacher_logits in batches_val_logits_teachers[batch_idx]]
                    for (head_logits, teacher_soft_targets) in zip(all_teachers_heads_logits, teachers_soft_targets):
                        distillation_losses_list.append(loss_function(F.log_softmax(head_logits,dim=1), teacher_soft_targets)*(1-ALPHA))
                distillation_losses_list = torch.tensor(distillation_losses_list, device=gpu)
                gt_head_loss = loss_function2(gt_head_val_soft_logits,labels)
                gt_head_loss = gt_head_loss.to(gpu)
                loss = torch.mean(distillation_losses_list) + gt_head_loss
                loss = loss.to(gpu)
            else:
                raise Exception(f"ERROR! expt should be \'student_MH_kd\', but got {expt}.")

            val_loss_list.append(loss.item())

    all_labels_val = torch.cat(labels_val_list)
    # Log ground truth head metrics and compute it's accuracy.
    gt_head_probs_val_list = [F.softmax(gt_head_logit, dim = 1) for gt_head_logit in gt_head_val_logits_list]
    gt_head_probs_val = torch.cat(gt_head_probs_val_list)
    gt_head_probs_val = gt_head_probs_val.reshape(gt_head_probs_val.shape[0]*gt_head_probs_val.shape[1],-1)
    gt_head__val_samples_certainties = get_samples_certainties(gt_head_probs_val, labels_val_list)
    _log_uncertainty("val ground truth head", gt_head__val_samples_certainties, epoch)
    _, gt_head_val_pred_final = torch.max(gt_head_probs_val, 1)
    gt_head_val_correct = (gt_head_val_pred_final==labels_val_list).sum().item()
    gt_head_val_acc = gt_head_val_correct / labels_val_list.size(0)
    # Log Mean and all heads, depends whether we aggregate or not.
    if hyper_params['aggregate_teachers']:
        mean_heads_probs_val = torch.cat(mean_heads_prob_val_list)
        mean_heads_probs_val = mean_heads_probs_val.reshape(mean_heads_probs_val.shape[0]*mean_heads_probs_val.shape[1], -1)
        mean_heads_val_samples_certainties = get_samples_certainties(mean_heads_probs_val, labels_val_list)
        _log_uncertainty("val teachers mean head", mean_heads_val_samples_certainties, epoch)
        _, mean_head_val_pred_final = torch.max(mean_heads_probs_val, 1)
        mean_head_val_correct = (mean_head_val_pred_final==labels_val_list).sum().item()
        mean_head_val_acc = mean_head_val_correct / labels_val_list.size(0)
    else:
        ## Below logging certainties of different heads.
        # Figured it out and probably works well.
        for name_head,head_prob_val_tensors in all_heads_prob_val_dict.items():
            all_heads_prob_val_dict[name_head] = torch.cat(head_prob_val_tensors)
        all_heads_val_certainties_dict = {name_head:get_samples_certainties(head_prob_val, labels_val_list) for \
                                            name_head,head_prob_val in all_heads_prob_val_dict.items()}
        for name_head,head_val_certainties in all_heads_val_certainties_dict.items():
            _log_uncertainty(f"val_{name_head}", head_val_certainties, epoch)
        all_heads_pred_val_dict_final = {name_head:(torch.max(head_prob_val_tensors, 1))[1] for \
                                        name_head,head_prob_val_tensors in all_heads_prob_val_dict.items()}
        all_heads_pred_val_correct = {f"{name_head}_correct":(head_pred_val_final==labels_val_list).sum().item() for \
                                        name_head,head_pred_val_final in all_heads_pred_val_dict_final.items()}
        all_heads_pred_val_acc = {f"{name_head}_acc":head_pred_val_correct/labels_val_list.size(0) for \
                                        name_head,head_pred_val_correct in all_heads_pred_val_correct.items()}

    if log_teacher_metrics and epoch==0:
        if hyper_params['aggregate_teachers']:
            ## Need to check this one!!
            all_teachers_val_prob_tensor = torch.stack(batches_val_logits_teachers, dim=0)
            all_teachers_val_prob_tensor = all_teachers_prob_tensor.reshape(all_teachers_val_prob_tensor.shape[0]*all_teachers_val_prob_tensor.shape[1], -1)
            mean_teachers_val_samples_certainties = get_samples_certainties(all_teachers_val_prob_tensor, labels_train_list)
            _, mean_teachers_val_pred = torch.max(all_teachers_val_prob_tensor, 1)
            mean_teachers_val_correct = (mean_teachers_val_pred==labels_val_list).sum().item()
            mean_teachers_val_acc = mean_teachers_val_correct / labels_val_list.size(0)
        else:
            all_teachers_val_prob_dict = {teacher_name:(torch.stack(teacher_prob_list, dim=0)) \
                                    for teacher_name,teacher_prob_list in teachers_pred_val_dict.items()}
            all_teachers_val_prob_dict = {teacher_name:(teach_prob_tensor.reshape(teach_prob_tensor.shape[0]*teach_prob_tensor.shape[1], -1)) \
                                    for teacher_name,teach_prob_tensor in all_teachers_val_prob_dict.items()}
            all_teachers_val_samples_certainties_dict = {teach_name:get_samples_certainties(teach_prob_val, labels_val_list) for \
                                                teach_name,teach_prob_val in all_teachers_val_prob_dict.items()}
            for teacher_name, teacher_probs in all_teachers_val_prob_dict.items():
                _, teacher_val_pred = torch.max(teacher_probs, 1)
                teacher_val_correct = (teacher_val_pred==labels_val_list).sum().item()
                teacher_val_acc = teacher_val_correct / labels_val_list.size(0)
                teachers_val_acc_dict[teacher_name] = teacher_train_acc

    # Below we log cerntainty and compute accuracy for mean_teachers or each teacher,
    # depends whether aggregate teachers or not.
    if hyper_params['aggregate_teachers']:
        _log_uncertainty(f"val_mean_teachers", mean_teachers_val_samples_certainties, epoch)
    else:
        for teacher_name, teacher_vall_amples_certainties in all_teachers_val_samples_certainties_dict.items():
            _log_uncertainty(f"val_teacher_{teacher_name}", teacher_vall_amples_certainties, epoch)

    assert 1==0, "Almost done!!"

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

    # Logging to wandb
    wandb.log({"train loss": train_loss, "val loss": val_loss, 'ground_truth_head_train_accuracy': gt_head_train_acc, \
                'ground_truth_head_val_accuracy': gt_head_val_acc, 'epoch':epoch})
    if hyper_params['aggregate_teachers']:
        wandb.log({'mean_teachers/mean_teachers_train accuracy': mean_teachers_train_acc, 'mean_teachers/mean_teachers_val accuracy': mean_teachers_val_acc, \
                    'epoch':epoch})
        wandb.log({'mean_heads/mean_teachers_head_train_accuracy': mean_head_train_acc , 'mean_heads/mean_teachers_head_val_accuracy': mean_head_val_acc ,\
                    'epoch':epoch})
    else:
        for teacher_name, teacher_train_acc in teachers_train_acc_dict.items():
            wandb.log({f'{teacher_name} teacher train accuracy': teacher_train_acc, 'epoch':epoch})
        for teacher_name, teacher_val_acc in teachers_val_acc_dict.items():
            wandb.log({f'{teacher_name} teacher val accuracy': teacher_val_acc, 'epoch':epoch})
        for head_name, head_train_acc in all_heads_pred_train_acc.items():
            wandb.log({f'{head_name}': head_train_acc, 'epoch':epoch})
        for head_name, head_val_acc in all_heads_pred_val_acc.items():
            wandb.log({f'{head_name}': head_val_acc, 'epoch':epoch})

    return  student_embedding, fc_heads, train_loss, val_loss, val_acc, max_val_acc,\
            batches_train_logits_teachers, batches_val_logits_teachers, all_teachers_samples_certainties_dict, \
            teachers_train_acc_dict, teachers_val_acc_dict, mean_teachers_train_acc

def get_samples_certainties(probs, labels):
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
