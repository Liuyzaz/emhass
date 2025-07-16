from setuptools import setup, find_packages

setup(
    name="emhass-optimization-analyzer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A general optimization analyzer for EMHASS optimization results.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/emhass-optimization-analyzer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "scikit-learn",  # Add any additional dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)