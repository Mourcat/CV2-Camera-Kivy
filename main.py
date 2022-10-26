import os

from pathlib import Path
BASE_DIR = Path.cwd()
IMG_DIR = BASE_DIR.joinpath('imgs')

import kivy
kivy.require('2.0.0')
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty, ObjectProperty, OptionProperty, StringProperty, BooleanProperty
from kivy.uix.image import Image

from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.dialog import MDDialog
from kivymd.toast import toast
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager

import cv2
import numpy as np


cascade_filepath_face = os.path.dirname(cv2.__file__)+ "/data/haarcascade_frontalface_alt2.xml"
cascade_filepath_eyes = os.path.dirname(cv2.__file__)+"/data/haarcascade_eye_tree_eyeglasses.xml"


class SettingsDialog(MDBoxLayout):
	pass


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
	
	imgFilename = StringProperty()
	cvImgToSave = None
	volume = None
		
	def __init__(self, **kwargs):
		super(CameraCV, self).__init__(**kwargs)
		self.capture = cv2.VideoCapture(self.camID)
		
		# Setting Camera properties 
		self.capture.set(
			cv2.CAP_PROP_FRAME_WIDTH,
			self.camWidth
		)
		
		self.capture.set(
			cv2.CAP_PROP_FRAME_HEIGHT,
			self.camHeight,
		)
		
		self.capture.set(
			cv2.CAP_PROP_EXPOSURE,
			1,
		)
		
		Clock.schedule_interval(
			self.cam_update,
			1/self.fps
		)
		self.show_properties()
		
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
		eyesClassifier= cv2.CascadeClassifier(cascade_filepath_eyes)
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
		eyes = eyesClassifier.detectMultiScale(
		    gray_image,
		    scaleFactor= 1.1,
		    minNeighbors= 5,
		    minSize=(10, 10),
		)
		faces_detected = format(len(faces)) + " faces detected!"
		eyes_detected = format(len(eyes)) + " eyes detected!"
		print(faces_detected)
		print(eyes_detected)
		
		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
		    cv2.rectangle(
		    	frame,
		    	(x, y),
		    	(x+w, y+h),
		    	color=(0, 255, 255),
		    	thickness=1,
		    )
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

	def show_properties(self):
		print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
		print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		print("CAP_PROP_FPS : '{}'".format(self.capture.get(cv2.CAP_PROP_FPS)))
		print("CAP_PROP_POS_MSEC : '{}'".format(self.capture.get(cv2.CAP_PROP_POS_MSEC)))
		print("CAP_PROP_FRAME_COUNT  : '{}'".format(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
		print("CAP_PROP_BRIGHTNESS : '{}'".format(self.capture.get(cv2.CAP_PROP_BRIGHTNESS)))
		print("CAP_PROP_CONTRAST : '{}'".format(self.capture.get(cv2.CAP_PROP_CONTRAST)))
		print("CAP_PROP_SATURATION : '{}'".format(self.capture.get(cv2.CAP_PROP_SATURATION)))
		print("CAP_PROP_HUE : '{}'".format(self.capture.get(cv2.CAP_PROP_HUE)))
		print("CAP_PROP_GAIN  : '{}'".format(self.capture.get(cv2.CAP_PROP_GAIN)))
		print("CAP_PROP_CONVERT_RGB : '{}'".format(self.capture.get(cv2.CAP_PROP_CONVERT_RGB)))
  

class InScreen(MDScreen):
	cvCam = ObjectProperty()
	dialog = None
	
	def setting_dialog(self):
		if not self.dialog:
			self.dialog = MDDialog(
				title='Settings',
				type='custom',
				content_cls=SettingsDialog(),
			)
			self.dialog.open()
	
	def on_enter(self, *args):
		animator = Animation(opacity=1)
		animator.start(self.ids.main_lay)
		self.cvCam = CameraCV(
			camID=0,
			camWidth=800,
			camHeight=600,
			fps=30,
		)
		self.ids.camera_lay.add_widget(self.cvCam)
		self.cvCam.update_names()


class UIApp(MDApp):
	ip = StringProperty()
	port = StringProperty()
	screens = []
	
	def build(self):
		self.theme_cls.primary_palette = 'Gray'
		self.theme_cls.accent_palette = 'Indigo'
		self.theme_cls.material_style = 'M3'
		self.theme_cls.theme_style = 'Dark'
		self.root = MDScreenManager()
		self.screens.append(InScreen(name='in_scr'))
		for scr in self.screens:
			self.root.add_widget(scr)
		self.root.current = 'in_scr'
		return self.root
		
	def on_start(self):
		pass
		
	def on_stop(self):
		self.root.screens[0].cvCam.capture.release()


if __name__ == '__main__':
	UIApp().run()