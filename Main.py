import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_ROOT = PROJECT_ROOT / 'yolov5'
RUNTIME_DIR = PROJECT_ROOT / '.runtime'
RUNTIME_DIR.mkdir(exist_ok=True)
# Keep third-party caches/settings inside the project so this works on machines
# where the user home directory is read-only (for example, sandboxes).
os.environ.setdefault('YOLOV5_CONFIG_DIR', str(RUNTIME_DIR / 'ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(RUNTIME_DIR / 'matplotlib'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, str(YOLO_ROOT))

ROOT = YOLO_ROOT

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from alert import alert_configuration_error, send_sms_alert
from classify import predict_species
from gradcam import make_gradcam_overlay
from threat import get_threat_tier, should_alert


@torch.no_grad()


def run(
        weights=PROJECT_ROOT / 'model/best.pt',  # model.pt path(s)
        source=PROJECT_ROOT / 'testImages/3.jpeg',  # file/dir/URL/glob, 0 for webcam
        data=YOLO_ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(196,196),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='yolov5\runs\detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        classify=True,  # identify each detected animal with the VGG-BiLSTM model
        gradcam=True,  # save Grad-CAM evidence overlays for classified detections
        send_alerts=False,  # send an SMS for the most confident species result
):
    detected_value = 0
    alert_candidates = []
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize =  False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # im.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f'{names[c]} {conf:.2f}'
                        if classify:
                            x1, y1, x2, y2 = [int(value) for value in xyxy]
                            height, width = im0.shape[:2]
                            x1, x2 = max(0, x1), min(width, x2)
                            y1, y2 = max(0, y1), min(height, y2)
                            crop = im0[y1:y2, x1:x2]
                            if crop.size:
                                try:
                                    if gradcam:
                                        overlay, species, species_conf = make_gradcam_overlay(crop)
                                        gradcam_dir = PROJECT_ROOT / 'gradcam'
                                        gradcam_dir.mkdir(exist_ok=True)
                                        cv2.imwrite(str(gradcam_dir / f'{p.stem}_{seen}_{c}.jpg'), overlay)
                                    else:
                                        species, species_conf = predict_species(crop)
                                    tier = get_threat_tier(species)
                                    if should_alert(species, species_conf):
                                        alert_candidates.append((species, species_conf, tier))
                                    label = species if hide_conf else f'{species} ({tier}) {species_conf * 100:.1f}%'
                                except (OSError, ValueError, RuntimeError) as exc:
                                    LOGGER.warning(f'Species classification skipped: {exc}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        detected_value = 1
                    #if save_crop:
                    #    save_one_box(xyxy, imc, file='output/test.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(str(PROJECT_ROOT / "output.png"), im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        if send_alerts and alert_candidates:
            species, confidence, _ = max(alert_candidates, key=lambda result: result[1])
            send_sms_alert(species, confidence)

        return detected_value


def main(image_path=PROJECT_ROOT / 'testImages/9.jpeg', classify=True, gradcam=True, send_alerts=False):
    """Run wildlife detection and return 1 when an animal is found, else 0."""
    check_requirements(exclude=('tensorboard', 'thop'))
    return run(source=image_path, classify=classify, gradcam=gradcam, send_alerts=send_alerts)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Detect wildlife in an image, video, folder, or camera stream.')
    parser.add_argument('source', nargs='?', default=PROJECT_ROOT / 'testImages/9.jpeg',
                        help='input image/video path, folder, URL, or camera index (default: testImages/9.jpeg)')
    parser.add_argument('--no-classify', action='store_true', help='skip second-stage species classification')
    parser.add_argument('--no-gradcam', action='store_true', help='skip Grad-CAM evidence-image generation')
    parser.add_argument('--send-alerts', action='store_true', help='send a Fast2SMS alert (requires environment variables)')
    parser.add_argument('--check-alert-config', action='store_true', help='verify .env SMS settings without sending a message')
    args = parser.parse_args()
    if args.check_alert_config:
        error = alert_configuration_error()
        print(f'Alert configuration: {error or "ready"}')
        raise SystemExit(1 if error else 0)
    print(main(args.source, classify=not args.no_classify, gradcam=not args.no_gradcam,
               send_alerts=args.send_alerts))
          
