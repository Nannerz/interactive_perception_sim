import pybullet as p
import pybullet_data
import sys
import threading
import os
import math
from pybullet_object_models import ycb_objects

# -----------------------------------------------------------------------------------------------------------


class Simulation:
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
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
            p.setGravity(0, 0, self.gravity)
            p.setRealTimeSimulation(1)
            p.setTimeStep(0.5/1000.0)
            p.setPhysicsEngineParameter(
                numSolverIterations=150,
                solverResidualThreshold=1e-9,
                frictionERP=0.10,  # smoother stick/slip
                useSplitImpulse=1,
                splitImpulsePenetrationThreshold=0.0005,
            )
            p.resetDebugVisualizerCamera(cameraDistance=1.2,
                                         cameraYaw=50,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0.5, 0, 0.2])
            p.loadURDF("plane.urdf")  # ground plane

            # urdf_dir = os.path.join(self.path, "panda_no_gripper.urdf")
            # self.robot = p.loadURDF(urdf_dir,
            self.robot = p.loadURDF("franka_panda/panda.urdf",
                                    basePosition=[0, 0, 0],
                                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True)

            for link in [9, 10]:
                p.changeDynamics(bodyUniqueId=self.robot, linkIndex=link,
                                 # contactStiffness=7.5e2,
                                #  contactStiffness=1e4,
                                 contactStiffness=5e3,
                                 # contactDamping=0.5,
                                 contactDamping=0.7,
                                #  collisionMargin=0.0005,
                                 lateralFriction=0.6,
                                #  rollingFriction=0.0005,
                                 spinningFriction=0.6)
                dyn = p.getDynamicsInfo(self.robot, link)
                print(f"Link {link} dynamics: {dyn}")
            num_joints = p.getNumJoints(self.robot)
            for i in range(0, num_joints):
                p.enableJointForceTorqueSensor(self.robot, i, 1)

        self.create_object()

    # -----------------------------------------------------------------------------------------------------------
    def create_object(self) -> None:
        flags = p.URDF_USE_INERTIA_FROM_FILE

        myobj = "cracker_box"
        # myobj = "mustard_bottle"
        # myobj = "pringles_can"

        if myobj == "mustard_bottle":
            base_orn = [0, 0, 20 * math.pi / 180]
            base_pos = [0.8, 0.04, 0.09]
            self.obj = self.create_mustard_bottle(flags, base_orn=base_orn, base_pos=base_pos)
        elif myobj == "pringles_can":
            base_orn = [0, 0, 0]
            base_pos = [0.8, 0.045, 0.03]
            # base_pos = [0.8, 0.06, 0.03]
            self.obj = self.create_pringles_can(flags, base_orn=base_orn, base_pos=base_pos)
        elif myobj == "cracker_box":
            # base_orn = [0, 0, 75 * math.pi / 180]
            # base_pos = [0.8, 0.025, 0.11]
            base_orn = [0, 0, 105 * math.pi / 180]
            base_pos = [0.8, 0.055, 0.11]
            # base_orn = [0, 0, 90 * math.pi / 180]
            # base_pos = [0.8, 0.04, 0.11]
            self.obj = self.create_cracker_box(flags, base_orn=base_orn, base_pos=base_pos)

    # -----------------------------------------------------------------------------------------------------------
    def create_mustard_bottle(self, flags, base_orn=[0, 0, 20 * math.pi / 180], base_pos=[0.8, 0.065, 0.08]) -> int:

        base_quat = p.getQuaternionFromEuler(base_orn)
        obj = p.loadURDF(
            os.path.join(ycb_objects.getDataPath(), "YcbMustardBottle", "model.urdf"),
            basePosition=base_pos,
            baseOrientation=base_quat,
            flags=flags)

        p.changeDynamics(
            bodyUniqueId=obj,
            linkIndex=-1,
            # contactStiffness=1e4,
            contactStiffness=7e4,
            # contactStiffness=1e4,
            # contactDamping=0.5,
            contactDamping=0.2,
            lateralFriction=0.4,
            spinningFriction=0.4)
        return obj

    # -----------------------------------------------------------------------------------------------------------
    def create_pringles_can(self, flags, base_orn=[0, 0, 20 * math.pi / 180], base_pos=[0.8, 0.065, 0.08]) -> int:

        base_quat = p.getQuaternionFromEuler(base_orn)
        obj = p.loadURDF(
            os.path.join(ycb_objects.getDataPath(), "YcbChipsCan", "model.urdf"),
            basePosition=base_pos,
            baseOrientation=base_quat,
            flags=flags)

        p.changeDynamics(bodyUniqueId=obj, linkIndex=-1,
                         # contactStiffness=1e4,
                         contactStiffness=6e4,
                         # contactStiffness=1e4,
                         # contactDamping=0.5,
                         contactDamping=0.5,
                         lateralFriction=0.35,
                         spinningFriction=0.35,)
        return obj

    # -----------------------------------------------------------------------------------------------------------
    def create_cracker_box(self, flags, base_orn=[0, 0, 20 * math.pi / 180], base_pos=[0.8, 0.065, 0.08]) -> int:

        base_quat = p.getQuaternionFromEuler(base_orn)
        obj = p.loadURDF(
            os.path.join(ycb_objects.getDataPath(), "YcbCrackerBox", "model.urdf"),
            basePosition=base_pos,
            baseOrientation=base_quat,
            flags=flags)

        p.changeDynamics(bodyUniqueId=obj, linkIndex=-1,
                         # contactStiffness=1e4,
                         contactStiffness=3e4,
                         # contactStiffness=1e4,
                         # contactDamping=0.5,
                         contactDamping=0.4,
                         lateralFriction=0.15,
                        #  rollingFriction=0.0005,
                         spinningFriction=0.15)

        return obj

    # -----------------------------------------------------------------------------------------------------------
    def create_red_cube(self) -> int:
        # create a red box at (0.5, 0, 0.05)
        # box_half_extents = [0.05]*3 # cube
        box_half_extents = [0.02, 0.02, 0.15]  # rectangle
        box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
        box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half_extents, rgbaColor=[1, 0, 0, 1])
        
        obj = p.createMultiBody(baseMass=1,
                                baseCollisionShapeIndex=box_col,
                                baseVisualShapeIndex=box_vis,
                                basePosition=[0.8, 0.0, 0.05])

        return obj

# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("This script should not be executed directly, exiting ...")
    sys.exit(0)
