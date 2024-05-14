from setuptools import setup

setup(
    name='blue_bird_pms',
    version='0.0.3.7',
    packages=['src/floor_ceiling_regime', 'src/regime', 'src/position_calculation'],
    url='',
    license='',
    author='bjahn',
    author_email='bjahnke71@gmail.com',
    description='',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'pandas_accessors @ git+https://github.com/bjahnke/pandas_accessors.git#egg=pandas_accessors',
    ]
)
