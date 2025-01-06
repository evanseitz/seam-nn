from setuptools import setup, find_packages

setup(
    name="seam-nn",
    version="0.1.1",
    author="Evan Seitz",
    author_email="evan.e.seitz@gmail.com",
    packages=find_packages(),
    description = "SEAM is a Python package to interpret sequence-based deep learning models for regulatory genomics data using meta-explanations.",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires=">=3.7.2",
    install_requires=[
    	'numpy',
	'matplotlib>=3.2.0',
	'pandas',
        'tqdm',
        'logomaker',
        'pyqt5',
        'psutil',
        'biopython',
        'tensorflow'
    ],
)