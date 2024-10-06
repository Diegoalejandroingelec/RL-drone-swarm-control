import cv2
import threading
import time
from apscheduler.schedulers.background import BackgroundScheduler
import time
import pygame
from datetime import datetime
import mediapipe as mp
from djitellopy import TelloSwarm
from real_time_agent_testing import initialize_agents, RL_image_classifier
import pickle
""" Initialization of variables"""
weights = 'v9_c_best.pt'
device = 0 # cuda device, i.e. 0 or 0,1,2,3 or cpu
imgsz=(640, 640)


# Global variable to signal movements
signal_takeoff=False
signal_up = False
signal_down=False
signal_left=False
signal_right=False
signal_land=False
signal_picture=False
signal_backward=False
signal_forward=False
is_rotating_clockwise = False
is_rotating_counter_clockwise = False
execution_counter=0
cmd = "_"
swarm="swarm_1"

original_frame = None
start_time = time.time()
# Create an instance of BackgroundScheduler
scheduler = BackgroundScheduler()

"""Pygame"""
pygame.init()
screen_width = 960
screen_height = 720
window = pygame.display.set_mode((screen_width,screen_height))
IMAGE_DIR = './assets/'
person = '-'

"""mediapipe"""
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()



my_drone = TelloSwarm.fromIps([
    # "192.168.0.2",
    "192.168.250.213"
    # "192.168.0.3",
])

my_drone_2 = TelloSwarm.fromIps([
    # "192.168.0.2",
    "192.168.250.237"
    # "192.168.0.3",
])



my_drone.connect()
my_drone_2.connect()

print('BATTERY-----------------------------------------------------------------')
for drone in my_drone:
    print(drone)
    print(drone.get_battery())

for drone in my_drone_2:
    print(drone)
    print(drone.get_battery())

print('BATTERY-----------------------------------------------------------------')


""" Face Recognition Embeddings"""
with open('core_embeddings.pkl', 'rb') as f:
    FaceEmbeddings = loaded_embeddings_dict = pickle.load(f)
    
FaceEncodings= FaceEmbeddings["FaceEmbeddings_new"]
FaceNames = FaceEmbeddings["FaceNames"]
print("Face Embedding loaded.")

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

def draw_face_and_hands(frame, face_landmarks, hand_landmarks):
    # Get the center of the frame (x-coordinate)
    frame_h, frame_w, _ = frame.shape
    frame_center_x = frame_w // 2
    frame_center_y = frame_h // 2
    
    h, w, c = frame.shape
    # Draw the face mesh and bounding box
    face_center_x = frame_center_x
    face_center_y = frame_center_y
    
    if face_landmarks:
        for landmarks in face_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
            # Determine the center of the face
            face_center_x = (x_max + x_min) // 2
            face_center_y = (y_max + y_min) // 2

            # Draw the center point of the face
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
            
            # Draw face bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            

            # mp_drawing.draw_landmarks(
            #     frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            #     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
            #     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
            # )

    # Draw the hands landmarks and bounding box
    if hand_landmarks:
        for landmarks in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
            # Draw hand bounding box
            #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # mp_drawing.draw_landmarks(
            #     frame, landmarks, mp_hands.HAND_CONNECTIONS,
            #     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            #     mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            # )


    return frame, face_center_x, face_center_y

def keep_alive(swarm):
    cmd = 'rc 0 0 0 0'
    for drone in swarm:
        drone.send_command_without_return(cmd)
    print(f"{swarm}, kept alive ‼️")

def get_flight_time():
    for drone in my_drone:
        flight_time = drone.get_flight_time()
    for done2 in my_drone_2:
        flight_time_2 = done2.get_flight_time()

    return flight_time, flight_time_2

def control_drone():
    global signal_takeoff
    global signal_up 
    global signal_down
    global signal_left
    global signal_right
    global signal_land
    global signal_backward
    global signal_forward
    global cmd
    global is_rotating_clockwise
    global is_rotating_counter_clockwise
    global original_frame
    global start_time
    global person
    global execution_counter
    global swarm


    
    while True:
        # ret, frame = cap.read()
        # # frame=my_drone.get_frame_read().frame
        # # ret = True
        # if ret:
        #     original_frame = frame.copy() 
        try:
            my_drone.parallel
            if signal_takeoff:
                print('Drone take off')
                flight_time, flight_time_2 = get_flight_time()
                print(flight_time)
                print(flight_time_2)
                if swarm == "swarm_1" and flight_time == 0:
                    my_drone.takeoff()
                    # Start the scheduler
                    scheduler.add_job(lambda: keep_alive(my_drone), 'interval', seconds=10)
                    # if len(scheduler.get_jobs()) > 1 and not scheduler.running:
                    scheduler.start()
                if swarm == "swarm_2" and flight_time_2==0:
                    my_drone_2.takeoff()
                    scheduler.add_job(lambda: keep_alive(my_drone_2), 'interval', seconds=10)
                    # if len(scheduler.get_jobs()) > 1 and not scheduler.running:
                    scheduler.start()

                signal_takeoff = False

                cmd = "_"
            if signal_up:
                print('Drone up')
                flight_time, flight_time_2 = get_flight_time()
                if swarm == "swarm_1" and flight_time != 0:
                    my_drone.move_up(20)
                if swarm == "swarm_2" and flight_time_2!=0:
                    my_drone_2.move_up(20)
                
                signal_up = False    
                cmd = "_"
            if signal_down:
                print('Drone down')
                flight_time, flight_time_2 = get_flight_time()
                if swarm == "swarm_1" and flight_time != 0:
                    my_drone.move_down(20)
                if swarm == "swarm_2" and flight_time_2!=0:
                    my_drone_2.move_down(20)
                
                signal_down = False   

                cmd = "_"
            if signal_left:
                print('Drone left')
                flight_time, flight_time_2 = get_flight_time()
                if swarm == "swarm_1" and flight_time != 0:
                    my_drone.move_left(20)
                if swarm == "swarm_2" and flight_time_2!=0:
                    my_drone_2.move_left(20)

                signal_left = False    
                    
                cmd = "_"
            if signal_right:
                print('Drone right')
                flight_time, flight_time_2 = get_flight_time()
                if swarm == "swarm_1" and flight_time != 0:
                    my_drone.move_right(20)
                if swarm == "swarm_2" and flight_time_2!=0:
                    my_drone_2.move_right(20)
                
                signal_right = False

                cmd = "_"
            if signal_backward:
                print('Drone back')
                flight_time, flight_time_2 = get_flight_time()
                if swarm == "swarm_1" and flight_time != 0:
                    my_drone.move_back(20)
                if swarm == "swarm_2" and flight_time_2!=0:
                    my_drone_2.move_back(20)

                signal_backward = False
                    
                cmd = "_"
            if signal_forward:
                print('Drone forward')
                flight_time, flight_time_2 = get_flight_time()
                if swarm == "swarm_1" and flight_time != 0:
                    my_drone.move_forward(20)
                if swarm == "swarm_2" and flight_time_2!=0:
                    my_drone_2.move_forward(20)

                signal_forward = False

                cmd = "_"
            if signal_land:
                print('Drone land')
                flight_time, flight_time_2 = get_flight_time()
                if swarm == "swarm_1" and flight_time != 0:
                    my_drone.land()
                if swarm == "swarm_2" and flight_time_2!=0:
                    my_drone_2.land()
                
                signal_land = False

                cmd = "_"
            if is_rotating_clockwise:
                print('Drone is rotating clock-wise')
                flight_time, flight_time_2 = get_flight_time()
                if swarm == "swarm_1" and flight_time != 0:
                    my_drone.my_drone.rotate_clockwise(20)
                if swarm == "swarm_2" and flight_time_2!=0:
                    my_drone_2.rotate_counter_clockwise(20)
                
                is_rotating_clockwise = False

                cmd = "_" 
            if is_rotating_counter_clockwise:
                print('Drone is rotating clock-wise')
                flight_time, flight_time_2 = get_flight_time()
                if swarm == "swarm_1" and flight_time != 0:
                    my_drone.my_drone.rotate_counter_clockwise(20)
                if swarm == "swarm_2" and flight_time_2!=0:
                    my_drone_2.rotate_clockwise(20)
                
                is_rotating_counter_clockwise = False

                cmd = "_" 
            time.sleep(0.1)  # Add a short delay to prevent this loop from consuming too much CPU
        except:
            print('FATAL ERROR TURN OFF YOUR COMPUTER NOW OR IT WILL GET DAMAGED!!')

"""  .....  """

cap = cv2.VideoCapture(0)
# my_drone = Tello()
# my_drone.connect() 
# my_drone.streamon()


# Start the get_frame thread
# get_frame_thread = threading.Thread(target=get_frame)
# get_frame_thread.daemon = True
# get_frame_thread.start()

# Start the control_drone thread
control_drone_thread = threading.Thread(target=control_drone)
control_drone_thread.daemon = True
control_drone_thread.start()



mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

flag = True
# Drone fligt
flying = False    


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

threshold = 5
previous_predicted_class = []
classes_counter = {'left':0,'right':0,'up':0,'down':0,'backwards':0,'forward':0,'land':0, 'no_action':0, 'take_off': 0}
swarm_classes_counter = {'swarm_1':0,'swarm_2':0,'no_action':0}
dqn_left,dqn_right = initialize_agents()

# with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
 mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        ret, frame = cap.read()
       
        # frame=my_drone.get_frame_read().frame
        ret = flag
        if ret:
            original_frame = frame.copy() 

            # cv2.imshow(" frame", frame)
            image = frame.copy()
            face_results = face_mesh.process(image)
            hand_results = hands.process(image)
            
                        
            """ get the face coordinates from mediapipe """
            frame, face_center_x, face_center_y = draw_face_and_hands(image, face_results.multi_face_landmarks, hand_results.multi_hand_landmarks)
            # cv2.imshow('Face and Hands Detection', image)
            # frame_rgb = image
            
            # Get the center of the frame (x-coordinate)
            frame_h, frame_w, _ = image.shape
            frame_center_x = frame_w // 2
            
            """ inferencing with yolo on 640x640 """
            # frame = apply_blur_except_regions(frame, regions)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                    
            predicted_class_left_hand, predicted_classes, annotated_frame = RL_image_classifier(hands, mp_drawing, mp_hands, original_frame, dqn_left, dqn_right)
            swarm = predicted_class_left_hand
            #print(f"Predicted class left hand: {predicted_class_left_hand} Predicted class right hand: {predicted_classes}")


            # predicted_classes,image = inference(image,model,names,line_thickness=3)
            # cv2.imshow("Detection frame", im0)
        
            # Draw the center line of the frame
            cv2.line(image, (frame_center_x, 0), (frame_center_x, frame_h), (255, 0, 0), 2)
            cv2.line(image, (frame_center_x + 120, 0), (frame_center_x + 120, frame_h), (0, 0, 255), 2)
            cv2.line(image, (frame_center_x - 120, 0), (frame_center_x - 120, frame_h), (0, 0, 255), 2)

            # Draw the center point of the face
            cv2.circle(image, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
            
            # Calculate the distance from the face center to the frame's center line
            x_distance = face_center_x - frame_center_x
            
            if abs(x_distance) >= 120:
                direction = "clockwise" if x_distance > 0 else "anti-clockwise"
                if flying:
                    
                    if direction == "clockwise":
                        cmd = "Rotating Clockwise"
                        is_rotating_clockwise = True
                    else:
                        cmd = "Rotating Counter-Clockwise"
                        is_rotating_counter_clockwise = True

            # Visualize the x distance
            cv2.line(image, (face_center_x, face_center_y), (frame_center_x, face_center_y), (250, 255, 0), 2)              
            # battery = my_drone.get_battery()
            battery = 100

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
            
            
            if(predicted_classes):
                
                classes_counter[predicted_classes]+=1
                swarm_classes_counter[predicted_class_left_hand]+=1

                if(previous_predicted_class and  predicted_classes!=previous_predicted_class):
                    classes_counter[previous_predicted_class]=0
                    swarm_classes_counter[predicted_class_left_hand]=0

                exceeding_threshold = {key: value for key, value in classes_counter.items() if value >= threshold}

                exceeding_threshold_2 = {key: value for key, value in swarm_classes_counter.items() if value >= threshold}

                is_empty = (not exceeding_threshold) or (not exceeding_threshold_2)
                if not is_empty:
                    #print(f'predicted class is: {exceeding_threshold}')
                    classes_counter =  {'left':0,'right':0,'up':0,'down':0,'backwards':0,'forward':0,'land':0, 'no_action':0, 'take_off': 0}
                    swarm_classes_counter = {'swarm_1':0,'swarm_2':0,'no_action':0}
                    class_action=list(exceeding_threshold.keys())[0]
                    
                    if class_action == 'up':
                        signal_up = True
                        cmd = "Move UP"
                    if class_action == 'left':
                        signal_right = True
                        cmd = "Move LEFT"
                    if class_action == 'down':
                        signal_down = True
                        cmd = "Move DOWN"
                    if class_action == 'right':
                        signal_left = True
                        cmd = "Move RIGHT"
                    if class_action == 'land':
                        signal_land = True
                        cmd = "LAND"
                    if class_action == 'backwards':
                        signal_backward=True
                        cmd = "BACK"
                    if class_action == 'forward':
                        signal_forward=True
                        cmd = "FORWARD"
                    if class_action == 'take_off':
                        signal_takeoff = True
                        cmd = "TAKE-OFF"
                        
                    class_action=''
                    
                previous_predicted_class=predicted_classes
                
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
