import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    
    model.eval()

    loss_list = []
    confusion = np.zeros((2, 2), dtype=int)
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            
            y_pred = out.squeeze(1).cpu().detach().numpy()
            y_true = msk.squeeze(1).cpu().detach().numpy()

            y_pred_binary = np.where(y_pred >= config.threshold, 1, 0)
            y_true_binary = np.where(y_true >= 0.5, 1, 0)

            confusion += confusion_matrix(y_true_binary.flatten(), y_pred_binary.flatten(), labels=[0, 1])

    if epoch % config.val_interval == 0:
        TN, FP, FN, TP = confusion.ravel()

        accuracy = (TN + TP) / np.sum(confusion) if np.sum(confusion) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, Iou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, ' \
                   f'specificity: {specificity}, sensitivity_or_Recall: {sensitivity}, percision：{precision}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()

    confusion = np.zeros((2, 2), dtype=int)

    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            y_pred = out.squeeze(1).cpu().numpy()
            y_true = msk.squeeze(1).cpu().numpy()
            
            if i % config.save_interval == 0:
                save_imgs(img, y_true, y_pred, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

            y_pred_binary = np.where(y_pred >= config.threshold, 1, 0)
            y_true_binary = np.where(y_true >= 0.5, 1, 0)

            confusion += confusion_matrix(y_true_binary.flatten(), y_pred_binary.flatten(), labels=[0, 1])


        TN, FP, FN, TP = confusion.ravel()

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
        
        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},Iou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, '\
                   f'specificity: {specificity}, sensitivity_or_Recall: {sensitivity}, percision：{precision}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)