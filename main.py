import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import tensorflow as tf

model = tf.keras.models.load_model("digit_model.keras")
print("AI Model Loaded Successfully!")

def predict_digit(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return -1, 0
    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y+h, x:x+w]
    size = max(w, h) + 40
    square = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = digit
    resized = cv2.resize(square, (28, 28))
    resized = resized.astype("float32") / 255.0
    resized = resized.reshape(1, 28, 28, 1)
    prediction = model.predict(resized, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return digit, confidence, prediction

def get_characters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    characters = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h < 20:
            characters.append((x, y, w, h, "-"))
        elif w > 15 and h > 15:
            characters.append((x, y, w, h, "digit"))
    characters = sorted(characters, key=lambda c: c[0])
    return characters

def rounded_rectangle(img, pt1, pt2, color, r=8):
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)

    cv2.circle(img, (x1+r, y1+r), r, color, -1)
    cv2.circle(img, (x2-r, y1+r), r, color, -1)
    cv2.circle(img, (x1+r, y2-r), r, color, -1)
    cv2.circle(img, (x2-r, y2-r), r, color, -1)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), np.uint8)
draw_color = (255, 255, 255)
save_requested = False
background_mode = "camera"
answer_text = ""
brush_size = 15
operator_menu = False
selected_operator = ""

first_number = ""
second_number = ""
waiting_for_second = False

prev_x, prev_y = None, None

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    print(img.shape)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                h, w, _ = img.shape
        x = int(handLms.landmark[8].x * w)
        y = int(handLms.landmark[8].y * h)
        
        index_tip = handLms.landmark[8].y
        index_pip = handLms.landmark[6].y
        middle_tip = handLms.landmark[12].y
        middle_pip = handLms.landmark[10].y

        mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.circle(img, (x, y), 10, draw_color, -1)
        if 10 < x < 90 and 10 < y < 50:
         draw_color = (255, 255, 255)

        elif 110 < x < 190 and 10 < y < 50:
            draw_color = (255, 150, 50)
        
        elif 210 < x < 290 and 10 < y < 50:
            draw_color = (0, 255, 0)
        
        elif 310 < x < 390 and 10 < y < 50:
            draw_color = (0, 255, 255)

        elif 410 < x < 490 and 10 < y < 50:
            draw_color = (180, 105, 255)
            
        elif 500 < x < 540 and 10 < y < 50:
            brush_size = 5

        elif 550 < x < 590 and 10 < y < 50:
            brush_size = 10

        elif 600 < x < 640 and 10 < y < 50:
            brush_size = 20
        elif 20 < x < 100 and 210 < y < 250:
            draw_color = (0, 0, 0)
        elif 20 < x < 100 and 270 < y < 310:
            if not operator_menu:
                operator_menu = True
            
        elif operator_menu and 110 < x < 150 and 270 < y < 310:
            selected_operator = "+"
            print("Selected:", selected_operator)
            operator_menu = False
            waiting_for_second = True
            
        elif operator_menu and 160 < x < 200 and 270 < y < 310:
            selected_operator = "-"
            print("Selected:", selected_operator)
            operator_menu = False
            waiting_for_second = True
            canvas = np.zeros((480,640,3),dtype=np.uint8)
            
        elif operator_menu and 210 < x < 250 and 270 < y < 310:
            selected_operator = "*"
            print("Selected:", selected_operator)
            operator_menu = False
            waiting_for_second = True
            canvas = np.zeros((480,640,3),dtype=np.uint8)
        
        elif operator_menu and 260 < x < 300 and 270 < y < 310:
            selected_operator = "/"
            print("Selected:", selected_operator)
            operator_menu = False
            waiting_for_second = True
            canvas = np.zeros((480,640,3),dtype=np.uint8)
               
        elif 20 < x < 100 and 90 < y < 130:
            background_mode = "camera"
            print("CAMERA MODE")
        
        elif 20 < x < 100 and 150 < y < 190:
            background_mode = "black"
            print("BLACK MODE")
        if index_tip < index_pip and middle_tip > middle_pip:
            if prev_x is not None and prev_y is not None:
                print(draw_color)
                print(prev_x, prev_y, x, y)
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_size)
                
                prev_x, prev_y = x, y
                
            else:
                prev_x, prev_y = None, None

        prev_x, prev_y = x, y
    else:
     prev_x, prev_y = None, None
    if background_mode == "black":
          black_bg = np.zeros((480, 640, 3), dtype=np.uint8)
          img = cv2.addWeighted(black_bg, 1, canvas, 1, 0)
    else:
     img = cv2.addWeighted(img, 1, canvas, 1, 0)
    
    cv2.rectangle(img, (10, 10), (90, 50), (255, 255, 255), -1)    # White
    cv2.rectangle(img, (110, 10), (190, 50), (255, 150, 50), -1)   # Blue
    cv2.rectangle(img, (210, 10), (290, 50), (0, 255, 0), -1)      # Green
    cv2.rectangle(img, (310, 10), (390, 50), (0, 255, 255), -1)    # Yellow
    cv2.rectangle(img, (410, 10), (490, 50), (180, 105, 255), -1)  # Pink
   # cv2.rectangle(img, (410, 60), (490, 100), (100, 100, 100), -1)
    
    #ERASER
    rounded_rectangle(img, (20, 210), (100, 250), (100, 100, 100))
    rounded_rectangle(img, (20, 210), (100, 250), (0, 0, 0), r=8)

    # Eraser body (coral)
    pts = np.array([[42, 230],
                [55, 217],
                [73, 235],
                [60, 248]], np.int32)

    cv2.fillPoly(img, [pts], (80, 127, 255))   # Coral (BGR)

# White stripe
    stripe = np.array([[55, 217],
                   [61, 223],
                   [48, 236],
                   [42, 230]], np.int32)
    cv2.fillPoly(img, [stripe], (255, 255, 255))

# Black outline
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    rounded_rectangle(img, (20, 90), (100, 130), (100, 100, 100))
    rounded_rectangle(img, (20, 90), (100, 130), (0, 0, 0), r=8)
    
# Camera body
    cv2.rectangle(img, (38, 100), (82, 125), (255, 255, 255), 2)

# Camera lens
    cv2.circle(img, (60, 112), 7, (255, 255, 255), 2)

# Top viewfinder
    cv2.rectangle(img, (45, 93), (58, 100), (255, 255, 255), -1)
   
    rounded_rectangle(img, (20, 150), (100, 190), (100, 100, 100))
    rounded_rectangle(img, (20, 150), (100, 190), (0, 0, 0), r=8)

    text = "DARK"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.55
    thickness = 2
    
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    
    x = 20 + (80 - text_w) // 2
    y = 150 + (40 + text_h) // 2 - 2
    
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness)
    
    # ---------------- Operator Button ----------------
    rounded_rectangle(img, (20, 270), (100, 310), (100, 100, 100))
    rounded_rectangle(img, (20, 270), (100, 310), (0, 0, 0), r=8)

    text = "OPS"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.6
    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)

    x = 20 + (80 - text_w) // 2
    y = 270 + (40 + text_h) // 2 - 2

    cv2.putText(img, text,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness)
    
    if operator_menu:
        rounded_rectangle(img, (110, 270), (150, 310), (80,80,80))
        rounded_rectangle(img, (160, 270), (200, 310), (80,80,80))
        rounded_rectangle(img, (210, 270), (250, 310), (80,80,80))
        rounded_rectangle(img, (260, 270), (300, 310), (80,80,80))

        cv2.putText(img, "+", (123,298),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.putText(img, "-", (175,298),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.putText(img, "*", (223,298),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.putText(img, "/", (273,298),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    #cv2.imshow("Canvas", canvas)
    
    rounded_rectangle(img, (503, 13), (537, 47), (150, 150, 150))
    #cv2.rectangle(img, (500, 10), (540, 50), (0, 0, 0), 1)
    rounded_rectangle(img, (553, 13), (587, 47), (150, 150, 150))
    rounded_rectangle(img, (603, 13), (637, 47), (150, 150, 150))
    
    cv2.putText(img, "S", (512, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2)

    cv2.putText(img, "M", (562, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2)

    cv2.putText(img, "L", (612, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2)
    
    if operator_menu or waiting_for_second:
        expression = first_number

        if selected_operator != "":
            expression += " " + selected_operator

        if second_number != "":
            expression += " " + second_number

        cv2.putText(
            img,
            "Expression: " + expression,
            (230, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
    if answer_text != "":
        cv2.putText(
            img,
            answer_text,
            (150, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    
    big_img = cv2.resize(img, (1000, 700))
    cv2.imshow("Air Drawing", big_img)

    key = cv2.waitKey(1) & 0xFF
    print(key)
    if key == ord('s'):
        save_requested = True
        print("waiting_for_second =", waiting_for_second)
    if save_requested:
        filename = datetime.now().strftime("drawing_%Y%m%d_%H%M%S.png")
        timestamp = datetime.now().strftime("%d-%m-%Y  %H:%M:%S")
        cv2.rectangle(
            img,
            (5, img.shape[0]-28),
            (175, img.shape[0]-5),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            img, 
            timestamp,
            (10, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA)
        cv2.imwrite(filename, img)
        print("Saved:", filename)
        
        characters = get_characters(canvas)
        result = ""
        for (x, y, w, h, char_type) in characters:
            char = canvas[y:y+h, x:x+w]
            digit, confidence, prediction = predict_digit(char)
            result += str(digit)

        if not waiting_for_second:
            first_number = result
            print("\n======================")
            print(" First Number : ", first_number)
            print("======================")
            
        else:
            second_number = result
            print("\n======================")
            print(" First Number : ", first_number)
            print(" Operator     : ", selected_operator)
            print(" Second Number: ", second_number)
            
        if first_number != "" and second_number != "":
            if selected_operator == "+":
                answer = int(first_number) + int(second_number)

            elif selected_operator == "-":
                    answer = int(first_number) - int(second_number)

            elif selected_operator == "*":
                    answer = int(first_number) * int(second_number)

            elif selected_operator == "/":
                answer = int(first_number) / int(second_number)

            print("----------------------")
            print(" Answer       : ", answer)
            print("======================")
            answer_text = "Answer: " + str(answer)

            first_number = str(answer)
            second_number = ""
            selected_operator = ""
            waiting_for_second = False
        save_requested = False
        if selected_operator != "":
            cv2.putText(img,
                "Operator: " + selected_operator,
                (170, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2)
    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), np.uint8)
  
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()