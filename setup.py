from setuptools import setup, version

with open("./README.md") as fh:
    long_description = fh.read()

with open("./requirements.txt") as fh:
    install_req = fh.read().strip().split("\n")

setup(
    name="graphpkg",
    version="0.0.1",
    description="This is a package for graphing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = "Nishant Baheti",
    author_email="nishantbaheti4764@gmail.com",
    packages = ["graphpkg"],
    url="https://github.com/NishantBaheti/graphpkg",
    project_urls={
        "Documentation": "https://nishantbaheti.github.io/graphpkg",
        "Bug Tracker": "https://github.com/NishantBaheti/graphpkg/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=install_req
)
