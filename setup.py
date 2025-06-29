"""
Setup script for SecPrompt package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="secprompt",
    version="0.1.0",
    author="SecPrompt Team",
    author_email="team@secprompt.com",
    description="A comprehensive framework for detecting and defending against prompt injection attacks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/secprompt/secprompt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "dashboard": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "secprompt=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "secprompt": ["data/*.json"],
    },
) 