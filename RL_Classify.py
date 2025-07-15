import cv2
import numpy as np
import mediapipe as mp
from real_time_agent_testing import initialize_agents, RL_image_classifier
import time

def draw_outlined_text(image, text, position, font, font_scale, color, thickness, outline_color, outline_thickness):
    """Draw text with outline for better visibility"""
    image = cv2.putText(image, text, position, font, font_scale, outline_color, outline_thickness, lineType=cv2.LINE_AA)
    image = cv2.putText(image, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return image

def clear_text_areas(image, areas):
    """Clear specific areas of the image by drawing black rectangles"""
    for area in areas:
        x, y, w, h = area
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return image

def test_hand_classification():
    """Test hand gesture classification without drone control"""
    
    # Initialize MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    # Initialize the RL agents
    print("Loading RL agents...")
    dqn_left, dqn_right = initialize_agents()
    print("RL agents loaded successfully!")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize counters for gesture tracking
    gesture_history = []
    max_history = 10
    frame_count = 0
    
    print("Starting hand gesture classification test...")
    print("Controls: 'q' to quit, 'r' to reset, 's' to save")
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as hands:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            try:
                # Get predictions from RL classifier (keep the annotated frame this time)
                predicted_class_left_hand, predicted_class_right_hand, annotated_frame = RL_image_classifier(
                    hands, mp_drawing, mp_hands, frame, dqn_left, dqn_right
                )
                
                # Clear the areas where the original text was placed to avoid overlap
                height, width = annotated_frame.shape[:2]
                text_areas_to_clear = [
                    (10, 10, 200, 60),  # Top left area where original text appears
                    (10, 35, 200, 25),  # Second line of original text
                ]
                annotated_frame = clear_text_areas(annotated_frame, text_areas_to_clear)
                
                # Store gesture history
                gesture_history.append({
                    'left': predicted_class_left_hand,
                    'right': predicted_class_right_hand,
                    'timestamp': time.time()
                })
                
                # Keep only recent history
                if len(gesture_history) > max_history:
                    gesture_history.pop(0)
                
                # Calculate gesture consistency
                if len(gesture_history) >= 5:
                    recent_left = [g['left'] for g in gesture_history[-5:]]
                    recent_right = [g['right'] for g in gesture_history[-5:]]
                    
                    left_consistency = recent_left.count(predicted_class_left_hand) / len(recent_left)
                    right_consistency = recent_right.count(predicted_class_right_hand) / len(recent_right)
                else:
                    left_consistency = 0.0
                    right_consistency = 0.0
                
                # Define text parameters
                font = cv2.FONT_HERSHEY_SIMPLEX
                title_font_scale = 0.7
                info_font_scale = 0.6
                small_font_scale = 0.5
                
                # Color definitions
                title_color = (255, 255, 255)
                info_color = (0, 255, 255)    # Cyan
                left_color = (255, 0, 255)    # Magenta
                right_color = (0, 255, 0)     # Green
                outline_color = (0, 0, 0)
                
                # TOP CENTER - Title
                title_text = "Hand Gesture Classification Test"
                (title_width, _), _ = cv2.getTextSize(title_text, font, title_font_scale, 2)
                title_x = (width - title_width) // 2
                annotated_frame = draw_outlined_text(
                    annotated_frame, title_text, 
                    (title_x, 30), font, title_font_scale, title_color, 2, outline_color, 4
                )
                
                # TOP RIGHT - Frame counter
                frame_text = f"Frame: {frame_count}"
                (frame_width, _), _ = cv2.getTextSize(frame_text, font, info_font_scale, 2)
                annotated_frame = draw_outlined_text(
                    annotated_frame, frame_text, 
                    (width - frame_width - 15, 30), font, info_font_scale, info_color, 2, outline_color, 3
                )
                
                # LEFT COLUMN - Left hand information
                left_column_x = 15
                left_y_start = 80
                line_spacing = 35
                
                annotated_frame = draw_outlined_text(
                    annotated_frame, f"Left Hand: {predicted_class_left_hand}", 
                    (left_column_x, left_y_start), font, info_font_scale, left_color, 2, outline_color, 3
                )
                
                annotated_frame = draw_outlined_text(
                    annotated_frame, f"Left Consistency: {left_consistency:.1%}", 
                    (left_column_x, left_y_start + line_spacing), font, small_font_scale, left_color, 1, outline_color, 2
                )
                
                # RIGHT COLUMN - Right hand information
                right_column_x = width // 2 + 20
                right_y_start = 80
                
                annotated_frame = draw_outlined_text(
                    annotated_frame, f"Right Hand: {predicted_class_right_hand}", 
                    (right_column_x, right_y_start), font, info_font_scale, right_color, 2, outline_color, 3
                )
                
                annotated_frame = draw_outlined_text(
                    annotated_frame, f"Right Consistency: {right_consistency:.1%}", 
                    (right_column_x, right_y_start + line_spacing), font, small_font_scale, right_color, 1, outline_color, 2
                )
                
                # BOTTOM ROW - Instructions (centered)
                instructions_text = "Press 'q' to quit, 'r' to reset, 's' to save"
                (instr_width, _), _ = cv2.getTextSize(instructions_text, font, small_font_scale, 1)
                instr_x = (width - instr_width) // 2
                annotated_frame = draw_outlined_text(
                    annotated_frame, instructions_text, 
                    (instr_x, height - 20), font, small_font_scale, (200, 200, 200), 1, outline_color, 2
                )
                
                # Display the frame
                cv2.imshow('Hand Gesture Classification Test', annotated_frame)
                
            except Exception as e:
                print(f"Error in classification: {e}")
                cv2.imshow('Hand Gesture Classification Test', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                gesture_history.clear()
                print("Gesture history reset")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"gesture_test_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Frame saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed!")

def analyze_gesture_performance():
    """Analyze gesture recognition performance over time"""
    
    # Initialize MediaPipe and agents
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    dqn_left, dqn_right = initialize_agents()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Performance tracking
    gesture_stats = {
        'left_hand': {'swarm_1': 0, 'swarm_2': 0, 'no_action': 0},
        'right_hand': {'up': 0, 'down': 0, 'left': 0, 'right': 0, 'backwards': 0, 'forward': 0, 'take_off': 0, 'land': 0, 'no_action': 0}
    }
    
    total_frames = 0
    
    print("Starting gesture performance analysis...")
    print("Press 'q' to quit and see final statistics")
    
    with mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            total_frames += 1
            
            try:
                # Get predictions and keep the annotated frame with hand landmarks
                predicted_class_left_hand, predicted_class_right_hand, annotated_frame = RL_image_classifier(
                    hands, mp_drawing, mp_hands, frame, dqn_left, dqn_right
                )
                
                # Clear the areas where the original text was placed
                height, width = annotated_frame.shape[:2]
                text_areas_to_clear = [
                    (10, 10, 200, 60),  # Top left area where original text appears
                ]
                annotated_frame = clear_text_areas(annotated_frame, text_areas_to_clear)
                
                # Update statistics
                gesture_stats['left_hand'][predicted_class_left_hand] += 1
                gesture_stats['right_hand'][predicted_class_right_hand] += 1
                
                # Define layout parameters
                font = cv2.FONT_HERSHEY_SIMPLEX
                title_font_scale = 0.7
                header_font_scale = 0.6
                stat_font_scale = 0.5
                
                # Color definitions
                title_color = (255, 255, 255)
                header_color = (0, 255, 255)
                left_color = (255, 0, 255)
                right_color = (0, 255, 0)
                outline_color = (0, 0, 0)
                
                # Title
                title_text = "Gesture Performance Analysis"
                (title_width, _), _ = cv2.getTextSize(title_text, font, title_font_scale, 2)
                title_x = (width - title_width) // 2
                annotated_frame = draw_outlined_text(
                    annotated_frame, title_text, 
                    (title_x, 30), font, title_font_scale, title_color, 2, outline_color, 4
                )
                
                # Total frames counter
                annotated_frame = draw_outlined_text(
                    annotated_frame, f"Total Frames: {total_frames}", 
                    (15, 70), font, header_font_scale, header_color, 2, outline_color, 3
                )
                
                # Left column - Left hand stats
                left_x = 15
                left_y_start = 110
                line_spacing = 25
                
                annotated_frame = draw_outlined_text(
                    annotated_frame, "Left Hand Stats:", 
                    (left_x, left_y_start), font, header_font_scale, left_color, 2, outline_color, 3
                )
                
                y_offset = left_y_start + 30
                for gesture, count in gesture_stats['left_hand'].items():
                    percentage = (count / total_frames) * 100 if total_frames > 0 else 0
                    annotated_frame = draw_outlined_text(
                        annotated_frame, f"{gesture}: {count} ({percentage:.1f}%)", 
                        (left_x + 10, y_offset), font, stat_font_scale, left_color, 1, outline_color, 2
                    )
                    y_offset += line_spacing
                
                # Right column - Right hand stats
                right_x = width // 2 + 20
                right_y_start = 110
                
                annotated_frame = draw_outlined_text(
                    annotated_frame, "Right Hand Stats:", 
                    (right_x, right_y_start), font, header_font_scale, right_color, 2, outline_color, 3
                )
                
                y_offset = right_y_start + 30
                for gesture, count in gesture_stats['right_hand'].items():
                    percentage = (count / total_frames) * 100 if total_frames > 0 else 0
                    annotated_frame = draw_outlined_text(
                        annotated_frame, f"{gesture}: {count} ({percentage:.1f}%)", 
                        (right_x + 10, y_offset), font, stat_font_scale, right_color, 1, outline_color, 2
                    )
                    y_offset += line_spacing
                
                # Bottom instructions
                instructions_text = "Press 'q' to quit and see final statistics"
                (instr_width, _), _ = cv2.getTextSize(instructions_text, font, stat_font_scale, 1)
                instr_x = (width - instr_width) // 2
                annotated_frame = draw_outlined_text(
                    annotated_frame, instructions_text, 
                    (instr_x, height - 20), font, stat_font_scale, (200, 200, 200), 1, outline_color, 2
                )
                
                cv2.imshow('Gesture Performance Analysis', annotated_frame)
                
            except Exception as e:
                print(f"Error: {e}")
                cv2.imshow('Gesture Performance Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Print final statistics
    print("\n=== GESTURE RECOGNITION STATISTICS ===")
    print(f"Total Frames Processed: {total_frames}")
    print("\nLeft Hand Gestures:")
    for gesture, count in gesture_stats['left_hand'].items():
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        print(f"  {gesture}: {count} frames ({percentage:.1f}%)")
    
    print("\nRight Hand Gestures:")
    for gesture, count in gesture_stats['right_hand'].items():
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        print(f"  {gesture}: {count} frames ({percentage:.1f}%)")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Hand Gesture Classification Tester")
    print("1. Basic Test")
    print("2. Performance Analysis")
    
    choice = input("Select option (1 or 2): ").strip()
    
    if choice == "1":
        test_hand_classification()
    elif choice == "2":
        analyze_gesture_performance()
    else:
        print("Invalid choice. Running basic test...")
        test_hand_classification()