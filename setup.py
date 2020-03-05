import os
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    'numpy>=1.16', 'reproject==0.5', 'scipy', 'astropy', 'ccdproc', 'astroscrappy', 'plotly'
]

os.system('pip install git+https://github.com/cylammarco/SpectRes')
os.system('pip install git+https://github.com/jveitchmichaelis/rascal.git@dev')

__packagename__ = "aspired"

setup(name=__packagename__,
      version='0.0.1',
      packages=find_packages(),
      author='Marco Lam',
      author_email='cylammarco@gmail.com',
      description="ASPIRED",
      url="https://github.com/cylammarco/ASPIRED",
      license='GPLv3',
      long_description=open('README.md').read(),
      zip_safe=False,
      include_package_data=True,
      install_requires=install_requires,
      python_requires='>=3.6')
