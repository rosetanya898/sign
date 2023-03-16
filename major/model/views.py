from django.shortcuts import render
from django.http.response import StreamingHttpResponse
#from .camera import  gen_frames
from .camera import SignLang
# Create your views here.


def index(request):
	return render(request, 'index.html')


def gen(camera):
	while True:
		frame = camera.gen_frames()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def mask_feed(request):
	return StreamingHttpResponse(gen(SignLang()),
					content_type='multipart/x-mixed-replace; boundary=frame')