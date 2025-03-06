"""
Setup configuration for the NeuralFlow application.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements/base.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neuralflow",
    version="0.1.0",
    author="Yavuz Topsever",
    author_email="yavuz.topsever@windowslive.com",
    description="Advanced LLM Workflow Orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neuralflow",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/neuralflow/issues",
        "Documentation": "https://github.com/yourusername/neuralflow/docs",
        "Source Code": "https://github.com/yourusername/neuralflow",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "typing-extensions>=4.5.0",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0.1",
        "aiohttp>=3.8.5",
        "asyncio>=3.4.3",
        "structlog>=23.1.0",
        "prometheus-client>=0.17.1",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "redis>=4.5.0",
        "aioredis>=2.0.0",
    ],
    extras_require={
        "dev": [
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.3.0",
            "pylint>=2.17.0",
            "sphinx>=6.2.1",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
            "pytest>=7.3.1",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.10.0",
            "pytest-xdist>=3.3.1",
            "pytest-benchmark>=4.0.0",
            "types-redis>=4.5.0",
            "types-PyYAML>=6.0.12",
            "types-python-dateutil>=2.8.19",
        ],
        "prod": [
            "gunicorn>=21.2.0",
            "uvicorn>=0.23.0",
            "fastapi>=0.100.0",
            "pydantic-settings>=2.0.0",
            "sentry-sdk>=1.29.0",
            "python-jose>=3.3.0",
            "passlib>=1.7.4",
            "bcrypt>=4.0.1",
            "ujson>=5.7.0",
            "orjson>=3.9.0",
            "msgpack>=1.0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuralflow=neuralflow.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "neuralflow": [
            "config/*.yaml",
            "config/*.json",
            "templates/*.html",
            "static/*",
        ],
    },
    zip_safe=False,
) 