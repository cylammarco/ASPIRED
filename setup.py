import os
from setuptools import setup, find_packages

install_requires=[
    'scipy', 'numpy>1.16', 'matplotlib', 'astropy', 'reproject==0.5', 'ccdproc', 'astroscrappy',
    'spectres @ git+https://github.com/cylammarco/SpectRes#egg=SpectRes', 'plotly'
    ]

os.system('pip install git+https://github.com/jveitchmichaelis/rascal.git')

__packagename__ = "aspired"

setup(
    name=__packagename__,
    version='0.0.1',
    packages=find_packages(),
    author='Marco Lam',
    author_email='cylammarco@gmail.com',
    license='GPLv3',
    long_description=open('README.md').read(),
    zip_safe=False,
    include_package_data=True,
    install_requires=install_requires
    )
