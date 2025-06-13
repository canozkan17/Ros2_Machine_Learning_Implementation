from setuptools import find_packages, setup

package_name = 'ros_ml_implementation'

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
    maintainer='can_ozkan',
    maintainer_email='can.ozkan.de@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "dataset_loader= ros_ml_implementation.dataset_loader:main",
            "preprocessor= ros_ml_implementation.preprocessor:main",
            "model_training= ros_ml_implementation.model_training:main",
            "model_test= ros_ml_implementation.model_test:main"
        ],
    },
)

