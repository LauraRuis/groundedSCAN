from setuptools import find_packages
from setuptools import setup

setup(name='GroundedScan',
      version='0.1',
      url='https://github.com/LauraRuis/groundedSCAN',
      author='Laura Ruis',
      author_email='lauraruis@live.nl',
      packages=find_packages(include=['GroundedScan']),
      install_requires=[
            'imageio~=2.9.0',
            'setuptools~=50.3.0',
            'gym~=0.17.2',
            'matplotlib~=3.3.2',
            'pronounceable~=0.1.3',
            'PyQt5~=5.15.1',
            'opencv-python~=4.1.2.30',
            'numpy~=1.19.2',
            'xlwt~=1.3.0',
      ]
)