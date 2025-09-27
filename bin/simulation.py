import pybullet as p # type: ignore
import pybullet_data # type: ignore
import sys
import threading
import os
from pybullet_object_models import ycb_objects # type: ignore
from typing import Any
p: Any = p

# -----------------------------------------------------------------------------------------------------------
class Simulation:
    def __init__(self) -> None:
        self.robot = None
        self.sim_lock = threading.Lock()
        self.gravity = -9.81
        self.obj = None

    # -----------------------------------------------------------------------------------------------------------
    def init_sim(self, sim_obj: dict[str, Any]) -> None:
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
                                 lateralFriction=0.7,
                                #  rollingFriction=0.0005,
                                 spinningFriction=0.7)

            num_joints = p.getNumJoints(self.robot)
            for i in range(0, num_joints):
                p.enableJointForceTorqueSensor(self.robot, i, 1)

        self.create_object(sim_obj)

    # -----------------------------------------------------------------------------------------------------------
    def create_object(self, sim_obj: dict[str, Any]) -> None:
        flags = p.URDF_USE_INERTIA_FROM_FILE
        match sim_obj['name']:
            case "mustard_bottle":
                self.obj = self.create_mustard_bottle(flags, base_orn=sim_obj['orn'], base_pos=sim_obj['pos'])
            case "pringles_can":
                self.obj = self.create_pringles_can(flags, base_orn=sim_obj['orn'], base_pos=sim_obj['pos'])
            case "cracker_box":
                self.obj = self.create_cracker_box(flags, base_orn=sim_obj['orn'], base_pos=sim_obj['pos'])
            case _:
                print("ERROR: Unknown object, exiting")
                sys.exit(1)

    # -----------------------------------------------------------------------------------------------------------
    def create_mustard_bottle(self, flags: int, base_orn: list[float], base_pos: list[float]) -> int:

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
    def create_pringles_can(self, flags: int, base_orn: list[float], base_pos: list[float]) -> int:

        base_quat = p.getQuaternionFromEuler(base_orn)
        obj = p.loadURDF(
            os.path.join(ycb_objects.getDataPath(), "YcbChipsCan", "model.urdf"),
            basePosition=base_pos,
            baseOrientation=base_quat,
            flags=flags)

        p.changeDynamics(bodyUniqueId=obj, linkIndex=-1,
                         # contactStiffness=9e4,
                         contactStiffness=9e4,
                         # contactStiffness=1e4,
                         # contactDamping=0.5,
                         contactDamping=0.9,
                         lateralFriction=0.35,
                         spinningFriction=0.35,
                         mass=1.0)
        return obj

    # -----------------------------------------------------------------------------------------------------------
    def create_cracker_box(self, flags: int, base_orn: list[float], base_pos: list[float]) -> int:

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
