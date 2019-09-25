from setuptools import setup, find_packages

install_requires=[
    'scipy', 'numpy', 'matplotlib', 'astropy', 'ccdproc', 'astroscrappy',
    'spectres @ git+https://github.com/cylammarco/SpectRes#egg=SpectRes', 'plotly'
    ]

__packagename__ = "aspired"

setup(
    name=__packagename__,
    version='0.0.1',
    packages=find_packages(),
    author='Marco Lam',
    author_email='cylammarco@gmail.com',
    license='GPL',
    long_description=open('README.md').read(),
    zip_safe=False,
    include_package_data=True,
    install_requires=install_requires
    )