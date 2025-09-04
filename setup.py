# setup.py
from setuptools import setup, find_packages

setup(
    name="medical-ai-system",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
)