from setuptools import find_packages, setup

package_name = 'body_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='devel',
    maintainer_email='timo.class@tum.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'standing = body_control.01_standing:main',
            'one_leg_stand = body_control.02_one_leg_stand:main',
            'squatting = body_control.03_squatting:main',
            't51 = body_control.t51:main',
            't52 = body_control.t52:main'
        ],
    },
)
