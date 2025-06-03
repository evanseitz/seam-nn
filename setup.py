from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="seam-nn",
    version="0.5.7",
    author="Evan Seitz",
    author_email="evan.e.seitz@gmail.com",
    description="SEAM: Meta-explanations for interpreting sequence-based deep learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evanseitz/seam-nn",
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires=">=3.7.2",
    install_requires=[
        'numpy',
        'matplotlib>=3.6.0',
        'pandas',
        'tqdm',
        'psutil',
        'biopython',
        'tensorflow>=2.0.0',
        'scipy>=1.7.0',
        'squid-nn',
        'seaborn'
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
        ],
        'dev': [
            'pytest',
            'pytest-cov',
            'flake8',
        ]
    }
)