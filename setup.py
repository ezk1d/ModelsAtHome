from setuptools import setup, find_packages

setup(
    name='ModelsAtHome',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'pandas'
    ],
)
