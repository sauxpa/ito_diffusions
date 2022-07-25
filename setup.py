from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ito_diffusions',
    packages=['ito_diffusions'],
    version='1.3.0',
    author="Patrick Saux",
    author_email="patrick.jr.saux@gmail.com",
    description="Library for stochastic process simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sauxpa/ito_diffusions",
    install_requires=['numpy', 'scipy', 'pandas', 'mpmath', 'tqdm'],
    python_requires='>=3.6',
)
