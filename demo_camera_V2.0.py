import argparse
import time
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.label_image as label_img
import copy
from tensorflow.keras.models import load_model
from PyQt5 import *
import PyQt5

#import tensorflow


fps_time = 0
minusX = 60
minusY = 100
H = 224
W = 224

# General Settings
prediction = ''
action = ''
score = 0
img_counter = 500


gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

model = load_model('models\VGG_cross_validated.h5')


def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score



threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter


def timeAlarm(array, flag):
    if len(humans) != 0 and flag == True:
        futureTime = time.time() + 10
        flag = False
        return futureTime, flag
    return futureTime, 

def cropHands(points, frame):
    # define croping values
    # output
    if (points[0] >= minusX and points[1] >= minusY):
        x = points[0] - minusX
        y = points[1] - minusY
    else:
        x = points[0]
        y = points[1]
    
        # Croping the frame
    crop_frame = frame[y:y+H, x:x+W]

    return crop_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    w, h = model_wh('0x0')
    if w > 0 and h > 0:
        e = TfPoseEstimator('models\graph\mobilenet_thin\graph_opt.pb', target_size=(w, h))
    else:
        e = TfPoseEstimator('models\graph\mobilenet_thin\graph_opt.pb', target_size=(432, 368))

    cam = cv2.VideoCapture(0)

    currentframe = 0
    flag = True
    is_okay = False
    while True:
    
        ret_val, image1 = cam.read()
        humans = e.inference(image1, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        img = TfPoseEstimator.draw_humans(image1, humans, imgcopy=True)
        
        if len(humans) != 0 and flag == True: # Time alarm
            future = time.time() + 25
            flag = False

        now = time.time()
        
        for human in humans:
            for i in human.body_parts.keys():
                
                if i == 7:
                    
                    left_wrist_point = human.body_parts[7]
                    image_h, image_w = image1.shape[:2]
                    center = (int(left_wrist_point.x * image_w + 0.5), int(left_wrist_point.y * image_h + 0.5))
                    
                    crop_frame = cropHands(center, image1)

                    cv2.imshow('frame',crop_frame)
                    
                    
                    #name = './images/frame' + str(currentframe) + '.jpg'
                    #print ('Creating...' + name) 
                    
                    # writing the extracted images 
                    
                    gray = cv2.cvtColor(crop_frame, cv2.COLOR_RGB2GRAY)
                    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
                    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    #cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    #    (255, 255, 255))
                    #cv2.putText(thresh, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    #    (255, 255, 255))

                    target = np.stack((thresh,) * 3, axis=-1)
                    target = cv2.resize(target, (224, 224))
                    target = target.reshape(1, 224, 224, 3)

                    prediction, score = predict_rgb_image_vgg(target)
                    
                    if score == 100 and prediction == 'Fist':
                        print("Fist_99")
                        print(score)
                    elif score > 95 and prediction == 'Okay':
                        is_okay = True
                        print("Okay_99")
                        print(score)
                    
                    elif score > 99 and prediction == 'L':
                        is_okay = True
                        print("L_99")
                        print(score)
                    #elif score > 99 and prediction == 'Palm':
                     #   print("Palm_99")
                     #   print(score)
                    #eliif score > 99 and prediction == 'Okay':
                     #   print("Okay_99")
                     #   print(score)

                elif now > future and (not is_okay) :
                    print("ALERT")
                    
                    #print(prediction)
                    #cv2.imwrite(name, gray) 
                    #currentframe += 1
                    # increasing counter so that it will 
                    # show how many frames are created 
                    
                    #result, score = predict(img)
                    #cv2.destroyAllWindows()
        
                
        cv2.imshow('tf-pose-estimation result', img)
        
        
        fps_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



     




# =============================================================================
# For running the script simply run the following in the cmd prompt/terminal :
# python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
# =============================================================================
