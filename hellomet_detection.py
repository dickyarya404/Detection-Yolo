import cv2
import numpy as np
from datetime import datetime
import threading as th
import schedule
import time
import pyodbc
import pyttsx3
#import library

#alert sound alarm suara pyttsx3 diinisialisasi untuk memutar suara alarm ketika orang terdeteksi tidak memakai helm
#Kecepatan suara diatur menjadi 150 kata per menit, dan suara default (male voice) dipilih
alarm_sound = pyttsx3.init()
voices = alarm_sound.getProperty('voices')
alarm_sound.setProperty('voice', voices[0].id)
alarm_sound.setProperty('rate', 150)

#yolov3 - tiny
net = cv2.dnn.readNet('yolov3_tiny_hellomet_v3.weights', 'yolov3_tiny_hellomet_v2.cfg')

classes = []
with open("helmet.names", "r") as f:
    classes = f.read().splitlines()

cv2.namedWindow("Hellomet - AIoT for Helmet Detection", cv2.WINDOW_NORMAL)
# Using resizeWindow()
cv2.resizeWindow("Hellomet - AIoT for Helmet Detection", 1000,800)

#cv2.namedWindow("Hellomet - AIoT for Helmet Detection")

#Input objek deteksi
cap = cv2.VideoCapture("testing/helmet_test10.mp4")
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("rtsp://10.9.0.152/live/ch00_1")
#cap.set(cv2.CAP_PROP_FPS, 30)

font = cv2.FONT_HERSHEY_SIMPLEX
colors = np.random.uniform(0, 255, size=(100, 3))


#database MY SQL connection
# db = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     passwd="root",
#     database="hellomet"
# )

# database SQL SERVER connection
# server = '10.5.0.123'
# database = 'hellomet'
# username = 'sa'
# password = 'Brekele893'
# db = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)


server = 'LAPTOP-9FK4RQP7'
database = 'hellomet'
username = 'admin'
password = 'admin'
db = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)

# datetime now today()
datetimenow = datetime.now()
str(datetimenow)
#now = datetimenow.today()

#function for insert to database
#Jumlah orang yang memakai helm (cw) dan tidak memakai helm (cnw) dihitung, lalu ditampilkan di layar.
def sctn():
    now = datetimenow.today()
    print(now)
    cursor = db.cursor()
    #sql = "INSERT INTO tb_dt_report (dt_report_datetime, dt_report_countusing, dt_report_countnotusing, dt_report_counttotal, dt_report_status) VALUES (%s,%s,%s,%s,%s)"
    sql = "INSERT INTO tb_dt_report (dt_report_datetime, dt_report_countusing, dt_report_countnotusing, dt_report_counttotal, dt_report_status) VALUES (?,?,?,?,?)"
    val = (now, cw, cnw, totalcount, status)
    cursor.execute(sql, val)
    db.commit()
    print("{} data ditambahkan".format(cursor.rowcount))

def voice_alarm(alarm_sound):
    try:
        alarm_sound.say("Not Wearing Helmet Detection")
        alarm_sound.runAndWait()

    except Exception as e:
        print(e)

schedule.every(10).seconds.do(sctn)
try:
    while True:
        _, img = cap.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []
        cnw = 0
        cw = 0
        totalcount = 0

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

                elif class_id == 1 and confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                # confidence = str(round(confidences[i], 2))
                confidence = str(int(confidences[i] * 100))
                color = colors[i]
                # cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence + "%", (x, y - 10), font, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                if label == 'Not Wearing Helmet':
                    cnw = cnw + 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    alarm = th.Thread(target=voice_alarm, args=(alarm_sound,))
                    alarm.start()

                elif label == 'Wearing Helmet':
                    cw = cw + 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                totalcount = cw + cnw

        # menampilkan text counting pada camera
        cv2.putText(img, "Wearing Helmet =  " + str(cw), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_4)

        cv2.putText(img, "Not Wearing Helmet = " + str(cnw), (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2, cv2.LINE_4)

        # statement status
        if cnw == 0 and cw != 0:  # safe
            status = int(0)
        elif cw != 0 and cnw > 0:  # warning
            status = int(1)
        elif cw == 0 and cnw != 0:  # danger
            status = int(2)

        cv2.imshow('Hellomet - AIoT for Helmet Detection', img)
        cv2.waitKey(1)
        schedule.run_pending()
        time.sleep(0)

        # key = cv2.waitKey(1)
        if cv2.waitKey(1) == 27 & 0xff:
            break

    alarm_sound.stop()
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(e)

except KeyboardInterrupt:
    print("Done")


