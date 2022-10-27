from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty, StringProperty, BooleanProperty
from kivymd.uix.behaviors import MagicBehavior, CircularElevationBehavior
from kivymd.toast import toast

import numpy as np
from settings import *


class CameraCV(Image):
	camID = NumericProperty()
	camWidth = NumericProperty()
	camHeight = NumericProperty()
	fps = NumericProperty()
	capture = None
	frame = None

	colorfmt = StringProperty()
	isCanny = BooleanProperty(False)
	isFaceDetecting = BooleanProperty(False)
	isEyesDetecting = BooleanProperty(False)
	
	exposure = NumericProperty(0)
	brightness  = NumericProperty(0)
	contrast = NumericProperty(0)
	hue = NumericProperty(0)
	saturation= NumericProperty(0)
	
	imgFilename = StringProperty()
	cvImgToSave = None
	volume = None
		
	def __init__(self, **kwargs):
		super(CameraCV, self).__init__(**kwargs)
		self.capture = cv2.VideoCapture(self.camID)
		self.capture.set(
			cv2.CAP_PROP_FRAME_WIDTH,
			self.camWidth
		)
		self.capture.set(
			cv2.CAP_PROP_FRAME_HEIGHT,
			self.camHeight,
		)
		'''
		self.capture.set(
			cv2.CAP_PROP_EXPOSURE,
			self.exposure,
		)
		self.capture.set(
			cv2.CAP_PROP_BRIGHTNESS,
			self.brightness
		)
		self.capture.set(
			cv2.CAP_PROP_CONTRAST,
			self.contrast,
		)
		self.capture.set(
			cv2.CAP_PROP_HUE,
			self.hue,
		)
		self.capture.set(
			cv2.CAP_PROP_SATURATION,
			self.saturation,
		)
		'''
		Clock.schedule_interval(
			self.cam_update,
			1/self.fps,
		)
		
	def cam_update(self, dt):
		ret, frame = self.capture.read()
		if ret:
			if self.isCanny:
				frame = cv2.cvtColor(
					frame,
					cv2.COLOR_RGB2BGR,
				)
				frame = self.canny_filter(frame)
				self.frame = frame
				self.colorfmt = 'luminance'
			elif self.isFaceDetecting:
				self.frame = self.detect_faces(frame)
				frame = cv2.cvtColor(
					self.frame,
					cv2.COLOR_RGB2BGR,
				)
			else:
				self.colorfmt = 'rgb'
				self.frame = frame
				frame = cv2.cvtColor(
					self.frame,
					cv2.COLOR_RGB2BGR,
				)
				
			prebuffer = cv2.flip(frame, 0)
			buffer = prebuffer.tobytes()
			
			img_texture = Texture.create(
				size=(
					frame.shape[1],
					frame.shape[0],
				),
				colorfmt=self.colorfmt,
			)
			img_texture.blit_buffer(
				buffer,
				colorfmt=self.colorfmt,
				bufferfmt='ubyte',
			)
			self.texture = img_texture
	
		
	

	def canny_filter(self, image):
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray_image, 100, 200)
		return edges

	def detect_faces(self, frame):
		faceClassifier = cv2.CascadeClassifier(cascade_filepath_face)
		gray_image = cv2.cvtColor(
			frame,
			cv2.COLOR_BGR2GRAY,
		)
		gray_image = cv2.equalizeHist(gray_image)
		faces = faceClassifier.detectMultiScale(
		    gray_image,
		    scaleFactor= 1.1,
		    minNeighbors= 5,
		    minSize=(10, 10),
		)
		faces_detected = format(len(faces)) + " faces detected!"
		print(faces_detected)
		for (x, y, w, h) in faces:
		    cv2.rectangle(
		    	frame,
		    	(x, y),
		    	(x+w, y+h),
		    	color=(0, 255, 255),
		    	thickness=1,
		    )
		return frame
		
	def detect_eyes(self, frame):
		eyesClassifier= cv2.CascadeClassifier(cascade_filepath_eyes)
		gray_image = cv2.cvtColor(
			frame,
			cv2.COLOR_BGR2GRAY,
		)
		gray_image = cv2.equalizeHist(gray_image)
		eyes = eyesClassifier.detectMultiScale(
		    gray_image,
		    scaleFactor= 1.1,
		    minNeighbors= 5,
		    minSize=(10, 10),
		)
		eyes_detected = format(len(eyes)) + " eyes detected!"
		print(eyes_detected)
		for (x, y, w, h) in eyes:
		    cv2.rectangle(
		    	frame,
		    	(x, y),
		    	(x+w, y+h),
		    	color=(0, 255, 255),
		    	thickness=1,
		    )
		return frame
	
	def take_pic(self):
		self.cvImgToSave = self.frame
		cv2.imwrite(
			f'{IMG_DIR}/{self.imgFilename}',
			self.cvImgToSave,
		)
		toast(f'{self.imgFilename} saved...')
		self.update_names()
		
	def update_names(self):
		self.volume = IMG_DIR.glob('*.png')
		self.imgFilename = f'img0{len(list(self.volume))+1}.png'

	def cam_destroy(self):
		if self.capture:
			Clock.unschedule(self.cam_update)
			self.capture.release()
			self.capture = None
			self.parent.remove_widget(self)

	def show_properties(self):
		print("CAP_PROP_FPS : '{}'".format(self.capture.get(cv2.CAP_PROP_FPS)))
		print("CAP_PROP_POS_MSEC : '{}'".format(self.capture.get(cv2.CAP_PROP_POS_MSEC)))
		print("CAP_PROP_FRAME_COUNT  : '{}'".format(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
		
		print("CAP_PROP_GAIN  : '{}'".format(self.capture.get(cv2.CAP_PROP_GAIN)))
		print("CAP_PROP_CONVERT_RGB : '{}'".format(self.capture.get(cv2.CAP_PROP_CONVERT_RGB)))