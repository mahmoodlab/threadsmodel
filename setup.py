from setuptools import setup, find_packages

setup(
    name='threadsmodel',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'einops',
        'huggingface_hub',
    ],
    author="Anurag Vaidya, Andrew Zhang, Guillaume Jaume",
    author_email="avaidya@mit.edu, andrewzh@mit.edu, gjaume@bwh.harvard.edu",
    description="A light-weight package for loading THREADS, a molecular-driven slide encoder for pathology (Nature Cancer, 2026).",
    url="https://github.com/mahmoodlab/threadsmodel",
    license="CC BY-NC-ND 4.0",
    include_package_data=True,
)
