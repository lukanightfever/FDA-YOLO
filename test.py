import argparse
import json
import os
from pathlib import Path
from threading import Thread

from models.yolo import Model
import numpy as np
import torch
import yaml
from tqdm import tqdm
from utils.datasets import Srotate, Nrotate
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import hbb_iou, clip_coords, coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr,torch_distributed_zero_first, \
    labels_to_class_weights
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel, intersect_dicts
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
import PIL.ImageDraw as ImageDraw
unloader = transforms.ToPILImage()
#yolov7
#def findpos(a, b):
    



def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.01,
         iou_thres=0.4,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=False,
         trace=False,
         is_coco=False,
         v5_metric=False,
         hyp = None):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Load model
        #with torch_distributed_zero_first(-1):
        #attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        nc = 4
        #nc = 15
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
        exclude = ['anchor'] if opt.cfg else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        hyp['cls'] *= nc / 80
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp
        imgsz = 1024
        gs = int(max(model.stride))
        with open(opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        path = data_dict['test']
        dataloader, dataset = create_dataloader(path, imgsz, batch_size, gs, opt, hyp=hyp, augment=False,
                                       cache=False, rect=True, local_rank=-1, world_size=1,shuffle=False)
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
        
        ema = ModelEMA(model)
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
        
        model = ema.ema
        
        #model.train()

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)


    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    #confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    
    print(names)
    
    
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    
    cccc = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        #print(half)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        '''
        #print(targets)
        imgs = img.clone()
        imgg = imgs[0]
        w=imgg.shape[1]
        h=imgg.shape[2]
        
        imgg = unloader(imgg)
        draw = ImageDraw.Draw(imgg)
        
        
        for lab in targets:
            label = lab.clone().unsqueeze(0)
            label[0][2] *= w
            label[0][4] *= w
            label[0][3] *= h
            label[0][5] *= h     
            cx = label[0][2].numpy().copy() #+ label[0][4].numpy()/4
            cy = label[0][3].numpy().copy() #+ label[0][5].numpy()/4
            label[:,2:]=xywh2xyxy(label[:,2:])
            label = label.squeeze(0)
            xmin, ymin, xmax, ymax, rota = label[2:]
            (left, right, top, bottom) = (xmin * 1, xmax * 1, ymin * 1, ymax * 1)
            x1,y1 = Nrotate(rota,left,top,cx,cy)
            x2,y2 = Nrotate(rota,left,bottom,cx,cy)
            x3,y3 = Nrotate(rota,right,bottom,cx,cy)
            x4,y4 = Nrotate(rota,right,top,cx,cy)
            x5,y5 = Nrotate(rota,left,top,cx,cy)
            draw.line([(x1, y1), (x2, y2), (x3, y3),(x4, y4), (x5, y5)], width=4, fill='OrangeRed')
        plt.figure(cccc)
        plt.imshow(imgg)
        f = plt.gcf()  #获取当前图像
        f.savefig(r'tempfig/{}.png'.format(cccc))
        cccc += 1     
        f.clear()  #释放内存
        continue
        '''
        
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height, 1.0]).to(device)
        #print(whwh)
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            #print(img.shape)
            out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss(train_out, targets.to(device))[1][:3]  # box, obj, cls

            # Run NMS
            t = time_synchronized()
            out = out.type(torch.float32)
            #print(conf_thres)
            #print(iou_thres)
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += time_synchronized() - t
        
        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            t_ang = labels[:, 5].tolist() if nl else []
            
            #ft = []
            #for i in range(nl):
            #    ft.append(0)
            ft = torch.zeros(nl) if nl else []
            seen += 1
            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls, t_ang, ft))
                continue
            
            clip_coords(pred, (height, width))
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            best_iou = torch.zeros(pred.shape[0], device=device)
            
            
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:6]) * whwh
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 6]).nonzero(as_tuple=False).view(-1)  # target indices
                    if pi.shape[0]:
                        ious, i = box_iou(pred[pi, :5], tbox[ti], device=device).max(1)  # best ious, indices
                        ii = 0
                        
                        for io in ious:
                        
                            if ious[ii]>= 0.5:
                                ft[ti[i[ii]]] = 1
                                
                            ii += 1
                        #print(ft[ti])
                        
                        best_iou[pi] = ious
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            #print(ft)

            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 6].cpu(), tcls, t_ang, ft))

        # Plot images
        #print(paths)
        
        #f = ('test_output/test_batch%g_gt.jpg' % batch_i)  # filename
        #plot_images(img, targets, paths, str(f), names)  # ground truth
        #f = ('test_output/test_batch%g_pred.jpg' % batch_i)
        #pt = output_to_target(out, width, height)     
        #plot_images(img, pt, paths, str(f), names)  # predictions
        
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        print(ap)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    #if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
    #    for i, c in enumerate(ap_class):
    #        print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    #if not training:
    #    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', type=str, default='runs/train/voc/weights/demo.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='models/yolov7-att.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/voc.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             hyp=hyp)

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
