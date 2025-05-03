import pybullet as p
import pybullet_data
import sys, threading, os

# -----------------------------------------------------------------------------------------------------------
class Simulation():
    _instance = None
    
    def __new__(cls, *args, **kwargs) -> 'Simulation':
        if cls._instance is None:
            cls._instance = super(Simulation, cls).__new__(cls)
        return cls._instance
# -----------------------------------------------------------------------------------------------------------
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.robot = None
        self.sim_lock = threading.Lock()
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.gravity = -9.81
# -----------------------------------------------------------------------------------------------------------
    def init_sim(self) -> None:
        with self.sim_lock:
            p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, self.gravity)
            p.setRealTimeSimulation(1)
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
            box_half_extents = [0.05]*3
            box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
            box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half_extents,
                                                rgbaColor=[1, 0, 0, 1])
            p.createMultiBody(baseMass=1,
                                    baseCollisionShapeIndex=box_col,
                                    baseVisualShapeIndex=box_vis,
                                    basePosition=[0.75, 0, 0.1])
# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("This script should not be executed directly, exiting ...")
    sys.exit(0)