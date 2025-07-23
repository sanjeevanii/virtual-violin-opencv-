import cv2
import mediapipe as mp
import time
import numpy as np
import pygame

pygame.mixer.init(frequency=44100, size=-16, channels=1)

def play_note(frequency, duration=0.5, volume=0.5):
    sample_rate = 44100
    n_samples = int(round(duration * sample_rate))
    buf = np.zeros((n_samples, 2), dtype=np.int16)
    max_sample = 2**15 - 1

    for s in range(n_samples):
        t = float(s) / sample_rate
        val = int(round(max_sample * np.sin(2 * np.pi * frequency * t)))
        buf[s][0] = val
        buf[s][1] = val
    sound = pygame.sndarray.make_sound(buf)
    sound.set_volume(volume)
    sound.play()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

zones = [100, 200, 300, 400]  

notes = [
    [196.00, 220.00, 246.94, 261.63, 293.66],   # G string
    [293.66, 329.63, 349.23, 392.00, 440.00],   # D string
    [440.00, 493.88, 523.25, 587.33, 659.25],   # A string
    [659.25, 698.46, 783.99, 880.00, 987.77]    # E string
]


last_play_time = 0
cooldown = 0.5  # seconds
prev_wrist = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for lm in handLms.landmark:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            wrist = lm_list[0]  


# Calculate speed 
            if prev_wrist is not None:
                dx = wrist[0] - prev_wrist[0]
                dy = wrist[1] - prev_wrist[1]
                speed = (dx**2 + dy**2)**0.5
                cv2.putText(frame, f"Bow speed: {int(speed)}", (400, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                bowing = speed > 20
            else:
                bowing = True  
            prev_wrist = wrist  

            x, y = lm_list[8]
           
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

            for i, zone in enumerate(zones):
                if zone < y < zone + 100:
                    fret_width = w // 5  
                    fret_index = min(x // fret_width, 4)  
                    note = notes[i][fret_index]
                    current_time = time.time()
                    if current_time - last_play_time > cooldown and bowing:
                        print(f"Playing: {note} Hz (String {i+1}, Fret {fret_index+1})")
                        play_note(note, duration=0.5)
                        last_play_time = current_time
                        break       

    # Draw zones
    for i, zone in enumerate(zones):
        cv2.rectangle(frame, (0, zone), (w, zone + 100), (0, 255, 0), 1)
        cv2.putText(frame, f"{notes[i]} Hz", (10, zone + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Virtual Violin", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()