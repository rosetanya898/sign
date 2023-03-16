import sys
import os
import threading
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from django.http import StreamingHttpResponse
import imutils
from imutils.video import VideoStream
import cv2

class SignLang:
    def __init__(self):
        self.cap = VideoStream(src=0).start()

    def __del__(self):
        cv2.destroyAllWindows()

    def predict(self,image_data,softmax_tensor,sess):
        predictions = sess.run(softmax_tensor, \
              {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        max_score = 0.0
        res = ''
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score > max_score:
                max_score = score
                res = human_string
        return res, max_score
  
    def gen_frames(self):
        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line
                        in tf.gfile.GFile("logs/trained_labels.txt")]
        # Unpersists graph from file
        with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            c = 0

            # Change the VideoCapture(0) argument to switch between camera sources
            #cap = VideoStream(src=0).start()

            res, score = '', 0.0
            i = 0
            mem = ''
            consecutive = 0
            sequence = ''

            while True:
                # Read a frame from the camera
                frame = self.cap.read()
                frame = cv2.flip(frame, 1)
                print(frame)

                # Crop the frame
                x1, y1, x2, y2 = 100, 100, 300, 300
                img_cropped = frame[y1:y2, x1:x2]
                priny(img_cropped)

                c += 1
                image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
                print(image_data)
                if i == 4:
                    res_tmp, score = self.predict(image_data,softmax_tensor,sess)
                    res = res_tmp
                    i = 0
                    if mem == res:
                        consecutive += 1
                    else:
                        consecutive = 0
                    if consecutive == 2 and res not in ['nothing']:
                        if res == 'space':
                            sequence += ' '
                        elif res == 'del':
                            sequence = sequence[:-1]
                        else:
                            sequence += res
                        consecutive = 0
                i += 1
                cv2.putText(frame, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
                cv2.putText(frame, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                mem = res
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

                # Create an image sequence showing the detected characters
                img_sequence = np.zeros((200,1200,3), np.uint8)
                cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                # Combine the frames and the image sequence
                frame = cv2.vconcat([frame, img_sequence])
                #print(frame)
                # Encode the frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                
                # Yield the frame as bytes
                return buffer.tobytes()