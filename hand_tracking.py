import cv2
import mediapipe as mp
import numpy as np

# -------- Camera --------
cap = None
for i in range(5):
    temp = cv2.VideoCapture(i)
    if temp.isOpened():
        cap = temp
        break

if cap is None:
    print("No camera found")
    exit()

# -------- MediaPipe --------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

tipIds = [4,8,12,16,20]

# -------- Canvas --------
canvas = np.zeros((480,640,3), dtype=np.uint8)
xp, yp = 0, 0
drawColor = (255,0,255)
brushThickness = 8
eraserThickness = 40

# -------- Colors --------
colors = [(255,0,255),(255,0,0),(0,255,0),(0,255,255)]
colorBoxes = [(0,0,160,60),(160,0,320,60),(320,0,480,60),(480,0,640,60)]

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img,1)
    img = cv2.resize(img,(640,480))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw color bar
    for i, box in enumerate(colorBoxes):
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),colors[i],cv2.FILLED)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList=[]
            for id,lm in enumerate(handLms.landmark):
                cx,cy=int(lm.x*640),int(lm.y*480)
                lmList.append((id,cx,cy))

            if lmList:
                fingers=[]

                # Thumb
                if lmList[4][1] > lmList[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other fingers
                for i in range(1,5):
                    if lmList[tipIds[i]][2] < lmList[tipIds[i]-2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                x1,y1 = lmList[8][1],lmList[8][2]

                # 🖐 Clear canvas
                if fingers.count(1) == 5:
                    canvas = np.zeros((480,640,3),dtype=np.uint8)

                # 🧽 Eraser (fist)
                if fingers.count(1) == 0:
                    if xp==0 and yp==0:
                        xp,yp=x1,y1
                    cv2.line(canvas,(xp,yp),(x1,y1),(0,0,0),eraserThickness)
                    xp,yp=x1,y1

                # ✌ Selection Mode
                elif fingers[1]==1 and fingers[2]==1:
                    xp,yp=0,0
                    if y1 < 60:
                        for i,box in enumerate(colorBoxes):
                            if box[0] < x1 < box[2]:
                                drawColor = colors[i]

                # ☝ Drawing Mode
                elif fingers[1]==1 and fingers[2]==0:
                    if xp==0 and yp==0:
                        xp,yp=x1,y1
                    cv2.line(canvas,(xp,yp),(x1,y1),drawColor,brushThickness)
                    xp,yp=x1,y1

            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    img = cv2.add(img,canvas)
    cv2.imshow("AI Air Canvas",img)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
