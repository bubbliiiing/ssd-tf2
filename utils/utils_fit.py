import os

import tensorflow as tf
from tqdm import tqdm


def get_train_step_fn():
    @tf.function
    def train_step(images, multiloss, targets, net, optimizer):
        with tf.GradientTape() as tape:
            prediction = net(images, training=True)
            loss_value = multiloss(targets, prediction)
            #------------------------------#
            #   添加上l2正则化参数
            #------------------------------#
            loss_value  = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step

@tf.function
def val_step(images, multiloss, targets, net):
    prediction = net(images)
    loss_value = multiloss(targets, prediction)
    #------------------------------#
    #   添加上l2正则化参数
    #------------------------------#
    loss_value  = tf.reduce_sum(net.losses) + loss_value
    return loss_value

def fit_one_epoch(net, multiloss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, save_period, save_dir):
    train_step  = get_train_step_fn()
    loss        = 0
    val_loss    = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            targets         = tf.convert_to_tensor(targets)
            
            loss_value      = train_step(images, multiloss, targets, net, optimizer)
            loss            = loss_value + loss

            pbar.set_postfix(**{'loss'  : float(loss) / (iteration + 1), 
                                'lr'    : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)
    print('Finish Train')

    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration>=epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            targets         = tf.convert_to_tensor(targets)

            loss_value      = val_step(images, multiloss, targets, net)
            val_loss        = val_loss + loss_value

            pbar.set_postfix(**{'loss' : float(val_loss)/ (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    logs = {'loss': loss.numpy() / (epoch_step+1), 'val_loss': val_loss.numpy() / (epoch_step_val+1)}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights(os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.h5" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
