import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="DeKAP",
    version="0.0.1",
    description="DeKAP demo program for paper Distillation-Enabled Knowledge Alignment Protocol for Semantic Communication in AI Agent Networks",
    author="Jingzhi Hu and Geoffrey Ye Li",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train_dynamic = source.distillation:main",
            "weight_sensitivity = z_paper_evaluations.eval_rev1_rd5:main",
        ],
    },
    install_requires=[],
    include_package_data=True,
)

