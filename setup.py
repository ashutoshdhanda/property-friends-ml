from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="property-friends-ml",
    version="1.0.0",
    author="Property Friends ML Team",
    description="ML Property Valuation API for Chilean Real Estate",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashutoshdhanda/property-friends-ml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "property-ml-train=src.models.train:main",
            "property-ml-api=src.api.main:start_server",
        ],
    },
)
