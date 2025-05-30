import pybullet as p
import pybullet_data
import sys, threading, os
# -----------------------------------------------------------------------------------------------------------
class Simulation():
    def __init__(self) -> None:
        self.robot = None
        self.sim_lock = threading.Lock()
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.gravity = -9.81
        self.obj = None
# -----------------------------------------------------------------------------------------------------------
    def init_sim(self) -> None:
        with self.sim_lock:
            p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, self.gravity)
            p.setRealTimeSimulation(0)
            p.resetDebugVisualizerCamera(cameraDistance=1.2,
                                            cameraYaw=50,
                                            cameraPitch=-30,
                                            cameraTargetPosition=[0.5, 0, 0.2])
            p.loadURDF("plane.urdf") # ground plane

            # urdf_dir = os.path.join(self.path, "panda_no_gripper.urdf")
            # self.robot = p.loadURDF(urdf_dir,
            self.robot = p.loadURDF("franka_panda/panda.urdf",
                                    basePosition=[0, 0, 0],
                                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True)
            num_joints = p.getNumJoints(self.robot)
            for i in range(0, num_joints):
                p.enableJointForceTorqueSensor(self.robot, i, 1)
        
        self.create_object()
# -----------------------------------------------------------------------------------------------------------
    def create_object(self) -> None:
        with self.sim_lock:
            # create a red box at (0.5, 0, 0.05)
            # box_half_extents = [0.05]*3 # cube
            box_half_extents = [0.02, 0.02, 0.15] # rectangle
            box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
            box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half_extents,
                                                rgbaColor=[1, 0, 0, 1])
            self.obj = p.createMultiBody(baseMass=1,
                                         baseCollisionShapeIndex=box_col,
                                         baseVisualShapeIndex=box_vis,
                                         basePosition=[0.8, 0.0, 0.05])
# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("This script should not be executed directly, exiting ...")
    sys.exit(0)