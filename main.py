import kivy
kivy.require('2.0.0')

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivymd.app import MDApp
from kivymd.uix.screenmanager import MDScreenManager

from views import InScreen


class AICameraApp(MDApp):
	ip = StringProperty()
	port = StringProperty()
	screens = []
	
	def build(self):
		self.theme_cls.primary_palette = 'Gray'
		self.theme_cls.accent_palette = 'Green'
		self.theme_cls.material_style = 'M3'
		self.theme_cls.theme_style = 'Dark'
		self.kvs_loader()
		self.root = MDScreenManager()
		self.screens.append(InScreen(name='in_scr'))
		for scr in self.screens:
			self.root.add_widget(scr)
		self.root.current = 'in_scr'
		return self.root
		
	def kvs_loader(self):
		Builder.load_file('views/screens/InScreen/in_screen.kv')
		Builder.load_file('views/components/CameraCV/cameracv.kv')
		
	def on_start(self):
		pass
		
	def on_stop(self):
		self.root.screens[0].cvCam.cam_destroy()


if __name__ == '__main__':
	AICameraApp().run()