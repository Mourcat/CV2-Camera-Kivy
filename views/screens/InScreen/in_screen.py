from kivy.animation import Animation
from kivy.clock import Clock
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.dialog import MDDialog
from kivymd.uix.screen import MDScreen

from views.components import CameraCV


class SettingsDialog(MDBoxLayout):
	pass
	

class InScreen(MDScreen):
	cvCam = None
	dialog = None
	
	def setting_dialog(self):
		if not self.dialog:
			self.dialog = MDDialog(
				title='Settings',
				type='custom',
				content_cls=SettingsDialog(),
			)
			self.dialog.open()
			
	def camera_start(self, cid, width, height, fps):
		self.cvCam = CameraCV(
			camID=cid,
			camWidth=width,
			camHeight=height,
			fps=fps,
		)
		self.ids.camera_lay.add_widget(self.cvCam)
		self.cvCam.update_names()
		
	
	def on_enter(self, *args):
		animator = Animation(opacity=1)
		animator.start(self.ids.main_lay)
		self.camera_start(0, 800, 600, 30)