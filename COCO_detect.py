import numpy as np
import torch
import cv2
from models.common import DetectMultiBackend
from utils.general import ( Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import pygame
import mediapipe as mp
pygame.init()


def load_model(weights,device,imgsz):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    return model,names,pt


classes_to_detect = ["person", "scissors"]

def inference(im,model,names,line_thickness):

    dt = (Profile(), Profile(), Profile())
    im=np.expand_dims(im,0)
    im = im[..., ::-1].transpose((0, 3, 1, 2))
    im0s=im.copy()
    im0s=np.squeeze(im0s,0)
    im0s= np.transpose(im0s, (1,2,0))
    
    # Initialize counter for persons
    object_count = 0
    
    with dt[0]:
        im = torch.from_numpy(im.copy()).to(model.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        
        pred = model(im, augment=False, visualize=False)
        pred = pred[0][1]

    # NMS
    with dt[2]:
        pred = non_max_suppression(prediction=pred,
                                    conf_thres=0.70,
                                    iou_thres=0.45,
                                    agnostic=False,
                                    max_det=1000)



    # Process predictions 
    det = pred[0]
    annotator = Annotator(np.ascontiguousarray(im0s), line_width=line_thickness, example=str(names))
    predicted_classes = []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            # Add bbox to image
            c = int(cls)  # integer class
            # label = f'{names[c]} {conf:.2f}'
            # annotator.box_label(xyxy, label, color=colors(c, True))
            # predicted_classes.append((names[c],conf.item()))
            if names[c] in classes_to_detect:
                label = f'{names[c]} {conf:.2f}'
                # if names[c] == "person":
                #     label = f'Puta {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                predicted_classes.append((names[c], conf.item()))
            
                # Count persons
                if names[c] == 'person':
                    object_count += 1

    # Stream results
    im0 = annotator.result()
    print(object_count)
    
    return predicted_classes,im0


def apply_blur_except_regions(frame, regions):
    # Apply a blur to the whole image
    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
    
    for region in regions:
        # Overlay the non-blurry region back onto the image
        x_min, y_min, x_max, y_max = region
        blurred_frame[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]
    
    return blurred_frame

def draw_VariableBoundingBox(frame, partLandmarks, color):
    h, w, _ = frame.shape
    regions = []  # To store bounding box coordinates
    for landmarks in partLandmarks:
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            if x > x_max:
                x_max = x + 60
            if y > y_max:
                y_max = y + 100
            if x < x_min:
                x_min = x - 60
            if y < y_min:
                y_min = y - 100
        regions.append((x_min, y_min, x_max, y_max))
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    return frame, regions

def draw_face_and_hands(frame, face_landmarks, hand_landmarks):
    regions = []  # To store all regions not to blur
    # Draw the face mesh and bounding box
    if face_landmarks:
        frame, face_regions = draw_VariableBoundingBox(frame, face_landmarks, (255, 255, 255))
        regions.extend(face_regions)

    # Draw the hands landmarks and bounding box
    if hand_landmarks:
        frame, hand_regions = draw_VariableBoundingBox(frame, hand_landmarks, (255, 0, 0))
        regions.extend(hand_regions)

    return frame, regions

def run(
        weights,
        device,
        imgsz,  # inference size (height, width)
        line_thickness=3,  # bounding box thickness (pixels)   
        ):

    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands


    model,names,_=load_model(weights,device,imgsz)
    cap = cv2.VideoCapture(0)

    window = pygame.display.set_mode((960,720))
    # classes_counter = {'left':0,'right':0,'up':0,'down':0,'backward':0,'forward':0,'land':0,'picture':0}
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, imgsz) 
                aux_image = frame.copy()
                face_results = face_mesh.process(aux_image)
                hand_results = hands.process(aux_image)
                _ , regions = draw_face_and_hands(aux_image, face_results.multi_face_landmarks, hand_results.multi_hand_landmarks)
                # frame = apply_blur_except_regions(frame, regions)

                predicted_classes,im0 =inference(frame,model,names,line_thickness)
                frame_surface = pygame.surfarray.make_surface(im0.swapaxes(0, 1))
                window.blit(frame_surface, (0, 0))
                pygame.display.update()

                cv2.waitKey(10)
       


if __name__ == "__main__":
    weights = 'v9_c_coco.pt'
    device = 0 # cuda device, i.e. 0 or 0,1,2,3 or cpu
    imgsz=(640, 640)
    run(weights,device,imgsz)
