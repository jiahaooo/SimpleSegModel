import os.path
import math
import argparse
import time
import random
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from utils import utils_image as util
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from tensorboardX import SummaryWriter

"""
###  Training  ###
"""

def main(json_path='options/train.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    print('\nprepare opt\n')

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path = option.find_last_checkpoint(opt['path']['models'])
    opt['path']['pretrained_network'] = init_path
    current_step = init_iter

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)
    if opt['sleep_time'] >= 1:
        print('sleep {:.2f} hours'.format(opt['sleep_time']/3600))
        time.sleep(opt['sleep_time'])

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    option.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))
    writer = SummaryWriter(os.path.join(opt['path']['log']))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = random.randint(1, 6249)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''
    print('\ncreat dataloader\n')

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,  # use or abandon the last minibatch
                                      pin_memory=False) # using swamp memory
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=False)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    print('\ninitialize model\n')
    model = define_Model(opt)

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    print('\nmain training\n')

    for epoch in range(opt['epoch_num']):  # keep running
        for i, train_data in enumerate(train_loader):
            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            # visual in tensorboard while training
            # report
            #       loss
            #       learning rate
            #       visual results
            # using logging and tensorboardX
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                    writer.add_scalar('{:s}'.format(k), v, current_step)
                logger.info(message)
                writer.add_scalar('lr', model.current_learning_rate(), current_step)
                # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

            if current_step == 1 or current_step % opt['train']['checkpoint_print'] == 0:
                training_visuals = model.current_results()
                TV_Mask = make_grid(training_visuals['Mask'].float(), nrow=2, normalize=True, scale_each=True)
                TV_Input = make_grid(training_visuals['Input'], nrow=2, normalize=True, scale_each=True)
                TV_Label = make_grid(training_visuals['Label'].float(), nrow=2, normalize=True, scale_each=True)
                writer.add_image('train - mask  image', TV_Mask, epoch)
                writer.add_image('train - input image', TV_Input, epoch)
                writer.add_image('train - label image', TV_Label, epoch)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:

                idx = 0
                avg_dice = 0.0
                avg_iou = 0.0

                for test_data in test_loader:
                    idx += 1
                    # -------------------------------
                    # 6-1) get test data
                    # -------------------------------
                    image_name_ext = os.path.basename(test_data['Input_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)
                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    # -------------------------------
                    # 6-2) do test
                    # -------------------------------
                    model.feed_data(test_data)
                    model.test()
                    visuals = model.current_visuals()

                    # -----------------------
                    # 6-3) save as .png files
                    # -----------------------
                    Mask_img = util.label_tensor2uint8(visuals['Mask'].float())
                    Label_img = util.label_tensor2uint8(visuals['Label'].float())
                    Input_img = util.tensor2uint8(visuals['Input'])
                    if current_step == opt['train']['checkpoint_test']:
                        util.imsave(Input_img, os.path.join(img_dir, 'Image_{:s}_{:d}.png'.format(img_name, current_step)))
                        util.imsave(Label_img, os.path.join(img_dir, 'Label_{:s}_{:d}.png'.format(img_name, current_step)))
                    util.imsave(Mask_img, os.path.join(img_dir, 'Mask_{:s}_{:d}.png'.format(img_name, current_step)))

                    # -----------------------
                    # 6-4) calculate indexes and report
                    # -----------------------
                    current_dice = util.caL_dice(visuals['Mask'], visuals['Label'])
                    current_iou = util.cal_iou(visuals['Mask'], visuals['Label'])
                    logger.info('{:->4d}--> {:>10s} | DICE {:<4.4f} | IOU {:<4.4f}'.format(idx, image_name_ext, current_dice, current_iou))
                    avg_dice += current_dice
                    avg_iou += current_iou

                avg_dice = avg_dice / idx
                avg_iou = avg_iou / idx

                # -----------------------
                # 6-5) report
                # -----------------------
                logger.info('<epoch:{:3d}, iter:{:8,d}, test finished'.format(epoch, current_step))
                writer.add_scalar('val_dice', avg_dice, current_step)
                writer.add_scalar('val_iou', avg_iou, current_step)
                # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
