setup(
    name='logomaker',
    version='0.8',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.16',
        'matplotlib>=3.1',
        'pandas>=0.24'
    ],
    extras_require={
        'gpu': ['tensorflow>=2.0.0'],
    }
) 