import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from nets.ssd_training import MultiboxLoss,Generator
from nets.ssd import SSD300
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from functools import partial
from tqdm import tqdm
import time

@tf.function
def train_step(imgs, multiloss, targets, net, optimizer):
    with tf.GradientTape() as tape:
        # 计算loss
        prediction = net(imgs, training=True)
        loss_value = multiloss(targets, prediction)

    grads = tape.gradient(loss_value, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss_value

@tf.function
def val_step(imgs, multiloss, targets, net, optimizer):
    # 计算loss
    prediction = net(imgs)
    loss_value = multiloss(targets, prediction)

    return loss_value

def fit_one_epoch(net, multiloss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, 
                Epoch):

    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_size:
                break
            images, targets = batch[0], batch[1]
            targets = tf.convert_to_tensor(targets)
            loss_value = train_step(images, multiloss, targets, net, optimizer)
            total_loss += loss_value

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss'        : float(total_loss) / (iteration + 1), 
                                'step/s'            : waste_time})
            pbar.update(1)
            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration>=epoch_size_val:
                break
            # 计算验证集loss
            images, targets = batch[0], batch[1]
            targets = tf.convert_to_tensor(targets)

            loss_value, _, _ = val_step(images, multiloss, targets, net, optimizer)
            # 更新验证集loss
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss)/ (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
      
#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    log_dir = "logs/"
    annotation_path = '2007_train.txt'
    
    NUM_CLASSES = 21
    input_shape = (300, 300, 3)
    priors = get_anchors()
    bbox_util = BBoxUtility(NUM_CLASSES, priors)
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('model_data/ssd_weights.h5', by_name=True, skip_mismatch=True)

    multiloss = MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #------------------------------------------------------#
    freeze_layer = 21
    for i in range(freeze_layer):
        model.layers[i].trainable = False
    #-------------------------------------#
    #   TF2的小bug，冻结后显存要求更大了
    #-------------------------------------#
    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        BATCH_SIZE = 4
        Lr = 5e-4
        Init_Epoch = 0
        Freeze_Epoch = 50

        generator = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                    (input_shape[0], input_shape[1]),NUM_CLASSES)

        if Use_Data_Loader:
            gen = partial(generator.generate, train = True)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
                
            gen_val = partial(generator.generate, train = False)
            gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=BATCH_SIZE).prefetch(buffer_size=BATCH_SIZE)
            gen_val = gen_val.shuffle(buffer_size=BATCH_SIZE).prefetch(buffer_size=BATCH_SIZE)

        else:
            gen = generator.generate(True)
            gen_val = generator.generate(False)

        epoch_size = num_train//BATCH_SIZE
        epoch_size_val = num_val//BATCH_SIZE
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Lr,
            decay_steps=epoch_size,
            decay_rate=0.9,
            staircase=True
        )

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(model, multiloss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, 
                        Freeze_Epoch)

    for i in range(freeze_layer):
        model.layers[i].trainable = True

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        BATCH_SIZE = 8
        Lr = 1e-4
        Freeze_Epoch = 50
        Epoch = 100

        generator =Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                    (input_shape[0], input_shape[1]),NUM_CLASSES)

        if Use_Data_Loader:
            gen = partial(generator.generate, train = True)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
                
            gen_val = partial(generator.generate, train = False)
            gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=BATCH_SIZE).prefetch(buffer_size=BATCH_SIZE)
            gen_val = gen_val.shuffle(buffer_size=BATCH_SIZE).prefetch(buffer_size=BATCH_SIZE)

        else:
            gen = generator.generate(True)
            gen_val = generator.generate(False)

        epoch_size = num_train//BATCH_SIZE
        epoch_size_val = num_val//BATCH_SIZE
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Lr,
            decay_steps=epoch_size,
            decay_rate=0.9,
            staircase=True
        )

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        for epoch in range(Freeze_Epoch,Epoch):
            fit_one_epoch(model, multiloss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, 
                        Epoch)

            
