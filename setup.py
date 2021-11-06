# Licensed under a 3-clause BSD style license - see LICENSE.rst

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_requires = ['numpy>=1.16', 'Cython>=0.29.23']

install_requires = [
    'scipy>=1.5.2', 'astropy>=4.0', 'ccdproc>=2.1',
    'plotly>=5.0', 'spectres>=2.1.1', 'statsmodels>=0.12',
    'astroscrappy==1.0.8', 'rascal==0.3.1'
]

__packagename__ = "aspired"

setup(name=__packagename__,
      version='0.4.1',
      packages=find_packages(),
      author='Marco Lam',
      author_email='cylammarco@gmail.com',
      description="ASPIRED",
      url="https://github.com/cylammarco/ASPIRED",
      license='bsd-3-clause',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      zip_safe=False,
      include_package_data=True,
      setup_requires=setup_requires,
      install_requires=install_requires,
      python_requires='>=3.6')
