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
    def init_sim(self, sim_obj: dict[str, Any] | None) -> None:
        with self.sim_lock:
            p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            # p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
            p.setGravity(0, 0, self.gravity)
            p.setRealTimeSimulation(1)
            # p.setTimeStep(0.5/1000.0)
            p.setPhysicsEngineParameter(
                numSolverIterations=50,
                solverResidualThreshold=1e-9,
                numSubSteps=5,
                # erp=0.2,
                # frictionERP=0.60,
                # contactERP=0.60,
                # allowedCcdPenetration=0.001,
                # warmStartingFactor=0.95,
                # contactBreakingThreshold=0.005,
                useSplitImpulse=0,
                # splitImpulsePenetrationThreshold=-0.005,
                # splitImpulsePenetrationThreshold=-0.02,
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
                p.changeDynamics(
                    bodyUniqueId=self.robot, 
                    linkIndex=link,
                    contactStiffness=15000.0,
                    contactDamping=20.0,
                    lateralFriction=0.7,
                    spinningFriction=0.1
                )

        if sim_obj is not None:
            self.create_object(sim_obj)

    # -----------------------------------------------------------------------------------------------------------
    def create_object(self, sim_obj: dict[str, Any]) -> None:
        objflags = p.URDF_USE_INERTIA_FROM_FILE
        objscale = 1.0
        objname = ""
        base_quat = p.getQuaternionFromEuler(sim_obj['orn'])
        base_pos = sim_obj['pos']
        dynargs: dict[str, Any] = {}
        
        match sim_obj['name']:
            # case "mustard_bottle":
            #     objname = "YcbMustardBottle"
            #     dynargs = {
            #         "contactStiffness": 70000,
            #         # contactStiffness=1e4,
            #         "contactDamping": 0.2,
            #         # contactDamping=0.5,
            #         #  rollingFriction=0.0005,
            #         "lateralFriction": 0.4,
            #         "spinningFriction": 0.4,
            #     }
            case "pringles_can":
                objname = "YcbChipsCan"
                objscale = 0.9
                dynargs = {
                    "contactStiffness": 4000.0,
                    "contactDamping": 20.0,
                    "lateralFriction": 0.45,
                    "spinningFriction": 0.1,
                    "mass": 0.6
                }
            case "cracker_box":
                objname = "YcbCrackerBox"
                dynargs = {
                    "contactStiffness": 4000.0,
                    "contactDamping": 20.0,
                    "lateralFriction": 0.4,
                    "spinningFriction": 0.1,
                    "mass": 0.411
                }
            case "meat_can":
                objname = "YcbPottedMeatCan"
                objscale = 1.3
                dynargs = {
                    "contactStiffness": 50000.0,
                    "contactDamping": 1.0,
                    "lateralFriction": 0.4,
                    "spinningFriction": 0.1,
                    "mass": 0.3
                }
            case "tomato_can":
                objname = "YcbTomatoSoupCan"
                objscale = 1.0
                dynargs = {
                    "contactStiffness": 50000.0,
                    "contactDamping": 1.0,
                    "lateralFriction": 0.45,
                    "spinningFriction": 0.1,
                    "mass": 1.0
                }
            case _:
                print("ERROR: Unknown object, exiting")
                sys.exit(1)

        try:
            obj = p.loadURDF(
                os.path.join(ycb_objects.getDataPath(), objname, "model.urdf"),
                basePosition=base_pos,
                baseOrientation=base_quat,
                globalScaling=objscale,
                flags=objflags)

            p.changeDynamics(bodyUniqueId=obj, 
                            linkIndex=-1,
                            **dynargs)
        except Exception as e:
            print(f"ERROR: Exception loading URDF object: {e}, exiting")
            sys.exit(1)

        self.obj = obj

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
