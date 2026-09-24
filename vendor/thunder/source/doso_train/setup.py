# Copyright (c) 2026 Shanghai Qiongpei Technology Co., Ltd.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Installation metadata for the private ``doso_train`` Isaac Lab extension."""

from pathlib import Path
import tomllib

from setuptools import find_packages, setup


EXTENSION_ROOT = Path(__file__).resolve().parent
with (EXTENSION_ROOT / "config" / "extension.toml").open("rb") as stream:
    EXTENSION = tomllib.load(stream)


setup(
    name="doso_train",
    version=EXTENSION["package"]["version"],
    description=EXTENSION["package"]["description"],
    author=EXTENSION["package"]["author"],
    maintainer=EXTENSION["package"]["maintainer"],
    url=EXTENSION["package"]["repository"],
    packages=find_packages(),
    package_data={
        "doso_train": [
            "data/Robots/DOSO/thunder/urdf/*.urdf",
            "data/Robots/DOSO/thunder/meshes/*.STL",
        ]
    },
    install_requires=["rsl-rl-lib==3.1.2", "cusrl[all]==1.1.9.post0"],
    python_requires=">=3.11,<3.12",
    license="Proprietary",
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Private :: Do Not Upload",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
)
