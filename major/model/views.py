from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
from django.shortcuts import render
import tensorflow as tf

def predict(image_data):

    predictions = sess.run(softmax_tensor, \
              {'DecodeJpeg/contents:0': image_data})
    print(predictions)
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    print(top_k)
    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

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
    softmax_tensor = tf.Session().graph.get_tensor_by_name('final_result:0')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.c = 0
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        res, score = '', 0.0
        i = 0
        mem = ''
        consecutive = 0
        sequence = ''
        while True:
            image = self.frame
            #print(image)
            imaged = cv2.flip(image, 1)
            #print(imaged)
            if self.grabbed:
                x1, y1, x2, y2 = 100, 100, 300, 300
                img_cropped = imaged[y1:y2, x1:x2]
                #print(img_cropped)
                self.c += 1
                #print(self.c)
                image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
                #print(image_data)
                if i == 4:
                    res_tmp, score = predict(image_data)
                    print(res_tmp, score)
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
                cv2.putText(image, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
                cv2.putText(image, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
                mem = res
                _, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def livefe(request):
    cam = VideoCamera()
    return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")


def index(request):
	return render(request, 'index.html')			
