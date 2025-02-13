from setuptools import setup, find_packages
import sys

# Ensure compatibility with Python 3.9-3.12
if sys.version_info < (3, 9) or sys.version_info >= (3, 13):
    sys.exit("snipe requires Python 3.9–3.12.")

setup(
    name="snipe",
    version="1.0.0",
    description="A tool for loading filterbank files, dedispersing, "\
            "removing RFI, and calculating SNR.",
    author="Wael Farah",
    author_email="wael.a.farah@gmail.com",
    packages=find_packages(),  # Automatically find all packages
    package_data={"snipe": ["help.txt"]},  # Include help.txt in the package
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "Pillow",  # PIL is in Pillow package
        "sigpyproc @ git+https://github.com/FRBs/sigpyproc3.git@v1.2.0"
    ],
    python_requires=">=3.9,<3.13",  # sigpyproc won't work after 3.12
    entry_points={
        "console_scripts": [
            "snipe=snipe.snipe:main",  # Allows `snipe` command in CLI
        ],
    },
)
