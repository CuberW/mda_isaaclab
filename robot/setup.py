"""Setup for robot project."""
from setuptools import setup, find_packages

setup(
    name="robot_unified",
    version="1.0.0",
    description="Unified robot system for three robot tasks",
    packages=find_packages(include=["robot_common", "robot_common.*",
                                     "task_319_garbage_sort",
                                     "task_22_dual_arm",
                                     "perception", "perception.*",
                                     "planning", "planning.*",
                                     "control", "control.*",
                                     "config", "config.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "mujoco>=3.1.0",
        "pyyaml>=6.0",
    ],
)
