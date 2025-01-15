import os
from PIL import Image
import numpy as np
import torch
# import random
import matplotlib.pyplot as plt
# from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from dataset.transforms import get_imagenet_transforms
# from matplotlib.ticker import ScalarFormatter

import pandas as pd
import monai
from monai.data import write_nifti
from monai.inferers import sliding_window_inference
from sklearn.metrics import roc_auc_score, confusion_matrix

import time
import math

outputs = []
def hook(module, input, output):
    global outputs
    outputs.clear()
    outputs.append(output)
    
def vis_feat(nrow, ncol, 
             vis_model, 
             target_layer, 
             input_list, 
             feat_idx=1, 
             test_dataset=None, 
             img_idx=None, 
             save_path=None, 
             vis_ouput=False,
             verbosity=False,
             num_feat=None,
):
    """
    Visualize the feature map
    
    """
    for i in range(nrow*ncol):
        
        if input_list is not None:
            input_img = input_list[i]
        else:
            input_img = test_dataset.__getitem__(img_idx[i])
            
        if 'layer' in target_layer:
            handle = getattr(vis_model, target_layer)[1].conv1.register_forward_hook(hook)
        else:
            handle = getattr(vis_model, target_layer).register_forward_hook(hook)
            
        output = vis_model(input_img)
        feat = outputs[0].detach().cpu().numpy().squeeze()
        print('Current subject {} with feat num {}.'.format(i, feat.shape[0]))
        
        # plot features
        if num_feat is None:
            total_num = feat.shape[0]
        else:
            total_num = num_feat
        len_row = 8
        # plt.figure('check', (20, 10))
        plt.figure('check', (20, 10*total_num/32))
        for j in range(total_num):
            img = feat[j]
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            # plt.subplot(1, total_num, j+1)
            plt.subplot(int(total_num/len_row), len_row, j+1)
            plt.title('Feat{}'.format(j))
            plt.imshow(img, cmap='viridis')
            plt.axis('off')
        plt.show()
            
        # plot only one feature
#         img = feat[feat_idx]
#         img = (img - np.min(img)) / (np.max(img) - np.min(img))
#         plt.subplot(nrow, ncol, i+1)
#         plt.imshow(img, cmap='viridis')
#         plt.axis('off')
        # plt.imsave(save_path.replace(".png", f"_{i}.png"), img, dpi=300, cmap='viridis')
        
        if verbosity:
            if i==range(nrow*ncol)[-1]:
                print('Current handle:')
                print(getattr(vis_model, target_layer))
                
                print('Image shape:')
                print('Output size:', output.shape, output.device)
                print('Outputs[0] size:', outputs[0].shape, outputs[0].device)
                print('Feature size:', feat.shape)
                
        handle.remove()
        del output


def run_inference_and_evaluate_2D(
    model, 
    batch, 
    roi_size, 
    plot_images=False, 
    save_output_path=None, 
    sw_batch_size=4,
    device=torch.device("cuda:0"),
    rot_degree=0,
):
    """Inference and evaluation function for segmentation.
    
    """
    
    model.to(device)
    
    # Use MONAI post-processing transforms
    argmax = monai.transforms.AsDiscrete(argmax=True)
    one_hot = monai.transforms.AsDiscrete(to_onehot=2)
    keep = monai.transforms.KeepLargestConnectedComponent(applied_labels=[1])
    
    # Quantify Dice, IOU, auc
    dice = monai.metrics.DiceMetric(include_background=False, reduction="mean", get_not_nans=False)            
    cal_iou = monai.metrics.MeanIoU(include_background=False, reduction="mean", get_not_nans=False)
    auc = monai.metrics.ROCAUCMetric()
    
    with torch.no_grad():        
        x = batch['INPUT'].to(device)
        y = batch['LABEL'].cpu()
        
        if rot_degree != 0:
            rot_trans = monai.transforms.Rotate(
                angle = math.radians(rot_degree),
                mode = 'nearest', # nearest / bilinear
                padding_mode='zeros',
            )
            x = rot_trans(img = x[0])[None]
            y = rot_trans(img = y[0])[None]
                
        start_time = time.time()
        y_pred = sliding_window_inference(x, roi_size, sw_batch_size, model)
        # y_pred = model(x)
        stop_time = time.time()
        
        inference_time = stop_time-start_time

        x = x.detach().cpu()
        y_pred = y_pred.detach().cpu()
        
        acc_list = list()
        sen_list = list()
        spe_list = list()
        iou_list =list()
        dice_list = list()
        # post_dice_list = list()
        auc_list = list()
        # post_auc_list = list()
        
        for i in range(y_pred.shape[0]):
            y_pred_i = y_pred[i] # CxHxW e.g. 2x512x512
            y_i = y[i]
            
            # print('pred shape: {}, label shape: {}; Dim: (C,H,W).'.format(y_pred_i.shape, y_i.shape))

            # Calculate IOU on the raw output
            # cal_iou(y_pred=one_hot(argmax(y_pred_i))[None], y=one_hot(y_i)[None])
            # iou_value = cal_iou.aggregate().item()
            # print('iou_value: ', iou_value)
            
            # Calculate AUC on the raw output
            y_auc      =               torch.transpose(torch.flatten(one_hot(y_i),start_dim=1),1,0)
            y_pred_auc = torch.softmax(torch.transpose(torch.flatten(y_pred_i,start_dim=1),1,0),dim=1)
            auc_value = roc_auc_score(y_true=y_auc, y_score=y_pred_auc)
            auc_list.append(auc_value)

            # Calculate confusion_matrix on the raw output
            [tn, fp, fn, tp] = confusion_matrix(y_true=torch.flatten(y_i), 
                                                y_pred=torch.flatten(argmax(y_pred_i))
                                                ).ravel()
            
            acc = (tp+tn) / (tn+fp+fn+tp)
            sen = tp / (tp+fn)
            spe = tn / (tn+fp)
            iou = tp / (tp+fp+fn)
            f1  = 2*tp / (2*tp+fp+fn)
            # print('acc: {}, sen: {}, spe: {}, iou: {}, f1: {},'.format(acc, sen, spe, iou, f1))
            acc_list.append(acc)
            sen_list.append(sen)
            spe_list.append(spe)
            iou_list.append(iou)
            dice_list.append(f1)
            
        
        df_eval = pd.DataFrame()
        df_eval['ROT'] = [rot_degree]
        df_eval['ACC'] = acc_list
        df_eval['SEN'] = sen_list
        df_eval['SPE'] = spe_list
        df_eval['IOU'] = iou_list
        df_eval['DICE'] = dice_list
        # df_eval['POST_DICE'] = post_dice_list
        df_eval['AUC'] = auc_list
        # df_eval['POST_AUC'] = post_auc_list
        df_eval['INFERENCE_TIME'] = inference_time
        # df_eval['DATASET_ID'] = batch['DATASET_ID'][0]
        df_eval['SUBJECT_ID'] = batch['SUBJECT_ID'][0]
        df_eval['INPUT'] = batch['INPUT_meta_dict']['filename_or_obj'][0]
        df_eval['LABEL'] = batch['LABEL_meta_dict']['filename_or_obj'][0]
        
        
        if plot_images:
            # print('image shape: {}, label shape: {}'.format(x.shape, y.shape))
            # print('pred shape: {}'.format(y_pred.shape))

            plt.figure('check', (12, 4))
            plt.subplot(1, 3, 1)
            plt.title('INPUT')
            plt.imshow(np.transpose(x[0], (1,2,0)))
            
            plt.subplot(1, 3, 2)
            plt.title('LABEL')
            plt.imshow(y[0,0,...])
            
            plt.subplot(1, 3, 3)
            plt.title('PRED')
            plt.imshow(argmax(y_pred[0,...])[0])
            plt.show()
    
        # Write data out
        if save_output_path is not None:
            # Get the image affine matrix
            current_affine = batch['IMAGE_meta_dict']['affine'][0].numpy()
            original_affine = batch['IMAGE_meta_dict']['original_affine'][0].numpy()
            original_spatial_shape = batch['IMAGE_meta_dict']['spatial_shape'][0].numpy()

            input_file_name = batch['IMAGE_meta_dict']['filename_or_obj'][0]
            output_file_name = os.path.split(input_file_name)[1]
            output_root_name = output_file_name[:-len('.nii.gz')]
            output_path = os.path.join(save_output_path,'{}_segm.nii.gz'.format(output_root_name))
            print('Saving segmentation results: {}'.format(output_path))
            
            df_eval['SEGM_RESULTS'] = output_path

            pred_output = argmax(y_pred[0,...]).numpy()
            
            writer = monai.data.NibabelWriter()
            writer.set_data_array(pred_output)
            writer.set_metadata(
                {
                    "affine": current_affine,
                    "original_affine": original_affine,
                }, 
                resample=False
            )
            writer.write(output_path, verbose=True)
                        
        return df_eval

