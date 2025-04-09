# setup.py

from setuptools import setup, find_packages

setup(
    name="yescarpenter",
    version="0.1.5",
    author="Shangcheng Zhao",
    author_email="shangchengzhao@gmail.com",
    description="A library providing frequently used functions in data analysis for YESlab members and other researchers",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/shangchengzhao/yescarpenter.git",  # update with your repo URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ],
)
