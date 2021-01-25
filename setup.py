from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_requires = ['numpy>=1.16']

install_requires = [
    'scipy>=1.5.2', 'astropy>=4.0,<4.3', 'ccdproc>=2.1', 'astroscrappy>=1.0.8',
    'plotly>=4.0', 'spectres>=2.1.1', 'rascal==0.2'
]

__packagename__ = "aspired"

setup(name=__packagename__,
      version='0.3.1',
      packages=find_packages(),
      author='Marco Lam',
      author_email='cylammarco@gmail.com',
      description="ASPIRED",
      url="https://github.com/cylammarco/ASPIRED",
      license='bsd-3-clause',
      long_description=open('README.md').read(),
      zip_safe=False,
      include_package_data=True,
      setup_requires=setup_requires,
      install_requires=install_requires,
      python_requires='>=3.6')
