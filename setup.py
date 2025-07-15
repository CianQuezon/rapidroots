# setup.py
from setuptools import setup, find_packages

setup(
    name="rapidroots",
    version="0.1.0",
    description="Symbolic function dispatcher",
    author="Shaun Quezon",
    author_email="you@example.com",
    python_requires=">=3.8",
    install_requires=[
        "pytest",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
