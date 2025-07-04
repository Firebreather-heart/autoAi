from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Base requirements
install_requires = [
    "pandas>=1.3.5",
    "numpy>=1.21.4",
    "scikit-learn>=1.0.1",
    "xgboost>=1.5.1",
    "imbalanced-learn>=0.8.0",
    "matplotlib>=3.4.3",
    "joblib>=1.1.0",
    "openpyxl>=3.0.9",
    "sqlalchemy>=1.4.27",
    "jinja2>=3.0.3",
]

# Optional dependencies
extras_require = {
    "deep_learning": ["tensorflow>=2.5.0"],
    "text": ["nltk>=3.6.5", "googletrans==4.0.0-rc1"],
    "image": ["opencv-python-headless>=4.5.4.60"],
    "web": ["flask>=2.0.0", "werkzeug>=2.0.0"],
    "full": [
        "tensorflow>=2.5.0",
        "nltk>=3.6.5",
        "googletrans==4.0.0-rc1",
        "opencv-python-headless>=4.5.4.60",
        "flask>=2.0.0",
        "werkzeug>=2.0.0",
    ]
}

setup(
    name="fireml",
    version="2.0.0",
    author="Akinrotimi Daniel Feyisola",
    author_email="dtenny95@gmail.com",
    description="An automated tool for data analysis, model training, and evaluation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/firebreather-heart/autoAI", # Replace with your repo URL
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'fireml=fireml.cli:main',
        ],
    },
)