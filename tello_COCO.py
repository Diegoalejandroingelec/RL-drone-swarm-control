from djitellopy import Tello
import cv2
import time
import pygame
from datetime import datetime

import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.general import ( Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


""" Initialization of variables"""
weights = 'v9_c_coco.pt'
device = 0 # cuda device, i.e. 0 or 0,1,2,3 or cpu
imgsz=(640, 640)


# Global variable to signal movements
cmd = "_"
original_frame = None
start_time = time.time()


"""Pygame"""
pygame.init()
screen_width = 960
screen_height = 720
window = pygame.display.set_mode((screen_width,screen_height))
IMAGE_DIR = './assets/'
person = '-'


def draw_outlined_text(image, text, position, font, font_scale, color, thickness, outline_color, outline_thickness):
    image = cv2.putText(image, text, position, font, font_scale, outline_color, outline_thickness, lineType=cv2.LINE_AA)
    image = cv2.putText(image, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return image


def picture_countdown_completed():
    global picture_counter
    global countdown_started_time
    if picture_counter == -1 or countdown_started_time is None:
        return False, False # Not finished, there is no countdown
    current_time = time.time()
    if current_time - countdown_started_time >= 1:
        countdown_started_time = time.time()
        picture_counter -= 1
        if picture_counter <= 0:
            picture_counter = -1
            countdown_started_time = None
            return True, True # Finished, there is a countdown
        return False, True # Not finished, there is a countdown
    
    return False, False # Not planned condition

def load_model(weights,device,imgsz):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    return model,names,pt




def inference(im,model,names,line_thickness):

    dt = (Profile(), Profile(), Profile())
    im=np.expand_dims(im,0)
    im = im[..., ::-1].transpose((0, 3, 1, 2))
    # im = im.transpose((0, 3, 1, 2))
    im0s=im.copy()
    im0s=np.squeeze(im0s,0)
    im0s= np.transpose(im0s, (1,2,0))
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
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
            predicted_classes.append((names[c],conf.item()))

    # Stream results
    im0 = annotator.result()
    
    return predicted_classes,im0

            


"""  .....  """

# cap = cv2.VideoCapture(0)
my_drone = Tello()
my_drone.connect() 
my_drone.streamon()


# Start the get_frame thread
# get_frame_thread = threading.Thread(target=get_frame)
# get_frame_thread.daemon = True
# get_frame_thread.start()


flag = False         # if robomaster
# Drone fligt
flying = False    


model,names,_=load_model(weights,device,imgsz)
# cap = cv2.VideoCapture(0)
# my_drone = Tello()
# my_drone.connect()
# my_drone.streamon()

"""PyGAME initialization/configuraiton"""
# Text attributes for buttons
button_images = [pygame.image.load(f'{IMAGE_DIR}up.png').convert_alpha(), 
                 pygame.image.load(f'{IMAGE_DIR}right.png').convert_alpha(), 
                 pygame.image.load(f'{IMAGE_DIR}down.png').convert_alpha(), 
                 pygame.image.load(f'{IMAGE_DIR}left.png').convert_alpha(), 
                 pygame.image.load(f'{IMAGE_DIR}picture.png').convert_alpha(), 
                 pygame.image.load(f'{IMAGE_DIR}backward.png').convert_alpha(), 
                 pygame.image.load(f'{IMAGE_DIR}take-off.png').convert_alpha(),
                 pygame.image.load(f'{IMAGE_DIR}land.png').convert_alpha(), 
                 pygame.image.load(f'{IMAGE_DIR}forward.png').convert_alpha(),
                 ]
frame_image = pygame.image.load(f'{IMAGE_DIR}frame.png')
frame_image = pygame.transform.scale(frame_image, (screen_width, screen_height))
    
button_rects = []
original_buttons = []
button_positions = [(820,510), (885, 575), (820, 640), (755, 575), (820, 575), (350,640), (415,640), (480, 640), (545, 640)]

for i in range(len(button_images)):
    button_images[i] = pygame.transform.scale(button_images[i], (65, 65))
    button_rects.append(button_images[i].get_rect(topleft=(button_positions[i])))
    original_buttons.append(button_images[i].copy())

# Alpha values for button transparency
default_alpha = 255  # Fully opaque
hover_alpha = 100  # Semi-transparent
click_alpha = 50   # More transparent on click

click_effect_duration = 200  # Duration of the click effect in milliseconds
last_click_time = 0  # Timestamp of the last click

window = pygame.display.set_mode((960,720))
""""""

threshold = 10
previous_predicted_class = []
classes_counter = {'left':0,'right':0,'up':0,'down':0,'backward':0,'forward':0,'land':0,'picture':0}


while True:
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # ret, frame = cap.read()
    frame=my_drone.get_frame_read().frame
    ret = flag
    if ret:
        original_frame = frame.copy() 
        frame = cv2.resize(frame, imgsz) 
        
        
        """ inferencing with yolo on 640x640 """
        # frame = apply_blur_except_regions(frame, regions)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predicted_classes,image = inference(image,model,names,line_thickness=3)
        # cv2.imshow("Detection frame", im0)
        
        battery = my_drone.get_battery()
        # battery = 100

        image=cv2.resize(image,(960,720))
        image = cv2.putText(image, f'Battery: {str(battery)} %', (760, 50), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 255), 1, cv2.LINE_AA)
        image = cv2.putText(image, f'Command: {cmd}', (15, 650), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 255), 1, cv2.LINE_AA)
        image = cv2.putText(image, f'Person: {person}', (15, 680), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        
        image=cv2.resize(image,(960,720))
        """PyGame window surface overlay"""
        
        frame_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        
        # Blit and display
        window.blit(frame_surface, (0, 0))
        window.blit(frame_image, (0, 0))
                        
        current_time_pygame = pygame.time.get_ticks()
        
        # Check for mouse hover
        mouse_pos = pygame.mouse.get_pos()
        
        for i in range(len(button_rects)):
            if button_rects[i].collidepoint(mouse_pos):
                if current_time_pygame - last_click_time > click_effect_duration:
                    # Change the alpha value when hovering
                    button_images[i].set_alpha(hover_alpha)
            else:
                # Reset the alpha value when not hovering or clicking
                if current_time_pygame - last_click_time > click_effect_duration:
                    button_images[i] = original_buttons[i].copy()
                    button_images[i].set_alpha(default_alpha)
                    
            window.blit(button_images[i], button_rects[i])
        pygame.display.update()
        """"""
        # cv2.waitKey(10)
        
        # Handle Pygame events
        for event in pygame.event.get():
             if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_q:
                     flag = False
                 elif event.key == pygame.K_w:
                     signal_up = True
                     cmd = "Move UP"
                 elif event.key == pygame.K_a:
                     signal_left = True
                     cmd = "Move LEFT"
                 elif event.key == pygame.K_s:
                     signal_down = True
                     cmd = "Move DOWN"
                 elif event.key == pygame.K_d:
                     signal_right = True
                     cmd = "Move RIGHT"
                 elif event.key == pygame.K_p:
                     frame_RGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     cv2.imwrite(f'./pictures/image_{i}_{current_time}.jpg', frame_RGB)
                     i += 1
                     signal_picture = True
                     cmd = "Take PICTURE"
                 elif event.key == pygame.K_t:
                     signal_takeoff = True
                     cmd = "TAKE-OFF"
                     flying = True
                 elif event.key == pygame.K_l:
                     signal_land = True
                     cmd = "LAND"
                 elif event.key == pygame.K_b:
                     signal_backward=True
                     cmd = "BACK"
                 elif event.key == pygame.K_f:
                     signal_forward=True
                     cmd = "FORWARD"
            # Mouse button down event
             elif event.type == pygame.MOUSEBUTTONDOWN:
                 for i in range(len(button_rects)):
                     if button_rects[i].collidepoint(event.pos):
                         button_images[i].set_alpha(click_alpha)
                         last_click_time = current_time_pygame
                         if i == 0:
                             signal_up = True
                             cmd = "Move UP"
                         elif i == 3:
                             signal_left = True
                             cmd = "Move LEFT"
                         elif i == 2:
                             signal_down = True
                             cmd = "Move DOWN"
                         elif i == 1:
                             signal_right = True
                             cmd = "Move RIGHT"
                         elif i == 4:
                             frame_RGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                             cv2.imwrite(f'./pictures/image_{i}_{current_time}.jpg', frame_RGB)
                             i += 1
                             picture_counter = 4
                             countdown_started_time = time.time()
                             print("TAKING PICTURE")
                             i += 1
                             signal_picture = True
                             cmd = "Take PICTURE"
                         elif i == 6:
                             signal_takeoff = True
                             cmd = "TAKE-OFF"
                             flying = True
                         elif i == 7:
                             signal_land = True
                             cmd = "LAND"
                         elif i == 5:
                             signal_backward=True
                             cmd = "BACK"
                         elif i == 8:
                             signal_forward=True
                             cmd = "FORWARD"
                                          
             elif event.type == pygame.QUIT:
                 flag = False
