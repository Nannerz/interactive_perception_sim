This code was created/tested with Python 3.12.1

The custom panda_with_sensor.urdf is just a copy of the pre-made panda.urd file from pybullet
The default can be found under the python install in: .venv\Lib\site-packages\pybullet_data\franka_panda

For this project to work, I made a symlink from the meshes folder to the base of the project
ln -sf .venv\Lib\site-packages\pybullet_data\franka_panda\meshes meshes
(Note that is a linux command with a windows path. Use git bash on windows with "/" instead of "\", or use the mklink command in cmd)