from launch import LaunchDescription
from launch.actions import ExecuteProcess
from pathlib import Path
import sys
import os
from launch.actions import IncludeLaunchDescription  # noqa: F401
from launch.launch_description_sources import PythonLaunchDescriptionSource  # noqa: F401
from launch_ros.actions import Node  # noqa: F401
from launch_ros.substitutions import FindPackageShare  # noqa: F401


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        ExecuteProcess(cmd=[str(Path(os.environ['CONDA_PREFIX'])/'bin'/'python') if os.getenv('CONDA_PREFIX') else sys.executable, '-m', 'custom.localizer_node'],
            cwd=str(Path(__file__).resolve().parent.parent),
            additional_env={'PYTHONPATH': str(Path(__file__).resolve().parent.parent)+os.pathsep+os.getenv('PYTHONPATH',''),
                            'PYTHONDONTWRITEBYTECODE':'1'}, output='screen'),
    ])
