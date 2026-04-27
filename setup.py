from setuptools import find_packages, setup


setup(
    name="rcwa3d",
    version="0.1.0",
    description="3D RCWA codebase split into rcwa3d_isotropic and rcwa3d_anisotropic packages.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "matplotlib>=3.5",
    ],
    extras_require={
        "gpu": ["torch>=2.0"],
    },
)
