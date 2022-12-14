import sys
import argparse

def get_args(description, mode='train'):
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('-s', '--seed', type=int, help = 'Give random seed')
    parser.add_argument('-d', '--dataset', choices = ['imagenette', 'imagewoof', 'cifar10'], help = 'Give the dataset name from the choices')
    if mode == 'train':
        parser.add_argument('-m', '--model', choices = ['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26', \
                            'resnet34', 'resnet34_0', 'resnet34_1', 'resnet34_2', 'resnet50', 'resnet101', 'vit_small', \
                            'vit_tiny', 'gernet_s', 'alexnet', 'resnet50_1', 'resnet50_2', 'resnet50_3', 'resnet50_4', 'resnet50_5', \
                            'resnet50_6', 'resnet50_7', 'resnet50_8', 'resnet50_9', 'resnet50_10',], \
                             help = 'Decide the student model architecture from the choices')
        parser.add_argument('-e', '--epoch', type=int, help = 'Give number of epochs for training')
        parser.add_argument('-p', '--percentage', type=int, help='Percentage of dataset to be used for training', default=None)
        parser.add_argument('-g', '--gpu', choices=[0, 1, 'cpu'], help='which GPU (or CPU) to be used for training', default=0)
        parser.add_argument('-a', '--api_key', type=str, help='Comet ML API key', default=None)
        parser.add_argument('-w', '--workspace', type=str, help='Comet ML workspace name or ID', default=None)
        parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate for training', default=None)
        parser.add_argument('-t', '--teacher', type=str, help='choose teacher or multiple teachers among available teachers. \
                                    If None then use pre-trained ResNet-34 as teacher \
                                    Note it requires teachers_num to be None.', default=None)
        parser.add_argument('-tn', '--teachers_num', type=int, help = 'Number of teachers for multiple teachers distillation', default=None)
        parser.add_argument('-ta', '--teacher_models', nargs="+", type=str, \
                            help = 'Teachers architecture to use for multipile teachers experiments. \
                                    Note it requires teachers_num to be greater than 1', default=None)
        parser.add_argument('-tt', '--teacher_training', action='store_true', help='Whether this experiment is for trainging a teacher' \
                            , default=False)
        parser.add_argument('-ud', '--update_teacher', action='store_true', help='In case we load a teacher, \
                             then load the teacher from wandb hub before using the local teacher' , default=False)
        parser.add_argument('--multi_head_fc_student', action='store_true', help='Train a student with multiple FC \
                             heads, each for every teacher, and additional one for the ground truth.' , default=False)
        parser.add_argument('--no_pre_trained_teacher', action='store_false' ,help='When loading a model to train \
                             as teacher, decide wether to fine-tune a pre-trained model or train from scratch. To load a pretrained model pass True. \
                             Note this should be used in "no-teacher" runs only.' , default=True)
        parser.add_argument('--aggregate_teachers', action='store_true' ,help=' Whether to aggregate teachers logits, or treate w.r.t each teacher. \
                             Only relevant in case of multi FC heads of a student, for each teacher, aka student_MH_kd expt! \
                            If not pass, treat each teacher seperately. If passed, average teachers.', default=False)


    elif mode == 'eval':
        parser.add_argument('-m', '--model', choices = ['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26'], help = 'Give the model name from the choices')
        parser.add_argument('-g', '--gpu', choices=[0, 1, 'cpu'], help='which GPU (or CPU) to be used for training', default=0)
        parser.add_argument('-p', '--percentage', type=int, help='Percentage of dataset to be used for training', default=None)
        parser.add_argument('-e', '--experiment', choices=['stagewise-kd', 'traditional-kd', 'simultaneous-kd', 'no-teacher', 'fsp-kd', 'attention-kd'])

    elif mode == 'data':
        parser.add_argument('-p', '--percentage', type=int, help='Give percentage of dataset')

    else: 
        sys.exit(f'invalid mode: {mode}')

    args = parser.parse_args()
    return args