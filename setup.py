from setuptools import setup, find_packages

with open('./requirements.txt', 'r') as f:
    packages = f.read().splitlines()

setup(
    name='Improving critical exponent estimations with Generative Adversarial Networks',
    packages=find_packages(),
    install_requires=packages,
)
