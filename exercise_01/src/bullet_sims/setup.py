from setuptools import find_packages, setup

package_name = 'bullet_sims'

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
            't2_temp = bullet_sims.t2_temp:run',
            't21 = bullet_sims.t21:run',
            't22 = bullet_sims.t22:run',
            't23 = bullet_sims.t23:main',
            't3_main = bullet_sims.t3_main:main',
            'teleoperation = bullet_sims.teleoperation:main'
        ],
    },
)
