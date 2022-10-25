from pathlib import Path
BASE_DIR = Path.cwd()
IMG_DIR = BASE_DIR.joinpath('imgs')

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty, ObjectProperty, OptionProperty, StringProperty, BooleanProperty
from kivy.uix.image import Image

from kivymd.app import MDApp
from kivymd.toast import toast
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager

import cv2
'''
haar_cascade_filepath = cv2.data.haarcascades + '/haarcascade_frontalface_default.xml'
import os
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)
'''
class CameraCV(Image):
	camID = NumericProperty()
	camWidth = NumericProperty()
	camHeight = NumericProperty()
	fps = NumericProperty()
	capture = None
	imgFilename = StringProperty()
	frame = None
	cvImgToSave = None
	volume = None
	colorfmt = StringProperty()
	isCanny = BooleanProperty(False)
	
	def __init__(self, **kwargs):
		super(CameraCV, self).__init__(**kwargs)
		#self.classifier = cv2.CascadeClassifier(cascade_filepath)
		self.capture = cv2.VideoCapture(self.camID)
		self.capture.set(
			cv2.CAP_PROP_FRAME_WIDTH,
			self.camWidth
		)
		
		self.capture.set(
			cv2.CAP_PROP_FRAME_HEIGHT,
			self.camHeight,
		)
		
		Clock.schedule_interval(
			self.cam_update,
			1/self.fps
		)
		
	def cam_update(self, dt):
		ret, frame = self.capture.read()
		if ret:
			self.frame = frame
			frame = cv2.cvtColor(
				self.frame,
				cv2.COLOR_RGB2BGR,
			)
			if self.isCanny:
				self.frame =self.canny_filter(frame)
				self.colorfmt = 'luminance'
			else:
				self.colorfmt = 'rgb'
				self.frame = frame

			prebuffer = cv2.flip(self.frame, 0)
			buffer = prebuffer.tobytes()
			
			img_texture = Texture.create(
				size=(
					self.frame.shape[1],
					self.frame.shape[0],
				),
				colorfmt=self.colorfmt,
			)
			img_texture.blit_buffer(
				buffer,
				colorfmt=self.colorfmt,
				bufferfmt='ubyte',
			)
			self.texture = img_texture
			
	def image_data_slot(self, image_data):
		if (self.width > self.height) != (image_data.shape[1] > image_data.shape[0]):
			# Need to rotate image data, the screen / camera is rotated
		    image_data = cv2.rotate(image_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
		    
		faces = self.detect_faces(image_data)
		for (x, y, w, h) in faces:
			cv2.rectangle(image_data, (x, y), (x + w, y + h), self._border, self._width)
		self.image = self.get_kvimage(image_data)
		#self.update()

	def get_kvimage(self, image):
		height, width, colors = image.shape
		image = Image(image.data, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
		return image

	def paintEvent(self, event):
		w = self.width()
		h = self.height()
		cw = self.image.width()
		ch = self.image.height()

		# Keep aspect ratio
		if ch != 0 and cw != 0:
			w = min(cw * h / ch, w)
			h = min(ch * w / cw, h)
			w, h = int(w), int(h)

		#painter.drawImage(QtCore.QRect(0, 0, w, h), self.image)
		self.image = Image()

	def canny_filter(self, image):
		gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray_image, 100, 200)
		return edges

	def detect_faces(self, image):
		gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray_image, 100, 200)

		gray_image = cv2.equalizeHist(gray_image)
		faces = self.classifier.detectMultiScale(gray_image, 1.3, 5)
		return faces
	
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


class InScreen(MDScreen):
	cvCam = ObjectProperty()
	
	def on_enter(self, *args):
		animator = Animation(opacity=1)
		animator.start(self.ids.cam_lay)
		
		self.cvCam = CameraCV(
			camID=0,
			camWidth=640,
			camHeight=480,
			fps=30,
		)
		self.ids.camera_lay.add_widget(self.cvCam)
		self.cvCam.update_names()


class UIApp(MDApp):
	ip = StringProperty()
	port = StringProperty()
	screens = []
	
	def build(self):
		self.theme_cls.primary_palette = 'Green'
		self.theme_cls.accent_palette = 'LightGreen'
		self.theme_cls.material_style = 'M3'
		self.theme_cls.theme_style = 'Dark'
		self.root = MDScreenManager()
		self.screens.append(InScreen(name='in_scr'))
		for scr in self.screens:
			self.root.add_widget(scr)
		self.root.current = 'in_scr'
		return self.root
		
	def on_stop(self):
		self.screens[0].cvCam.capture.release()

if __name__ == '__main__':
	UIApp().run()