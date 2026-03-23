from setuptools import setup
import sys

setup(
    name="WFlib",
    version="0.1",
    description="Library for website fingerprinting attacks",
    author="Xinhao Deng",
    packages=["WFlib", "WFlib.models", "WFlib.tools"],
    install_requires=["tqdm", "numpy", "pandas", "scikit-learn", "einops", "timm", "pytorch-metric-learning", "captum"],
)
