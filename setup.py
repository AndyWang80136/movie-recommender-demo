from pathlib import Path

from setuptools import find_packages, setup


def parse_requirement(requirement_file: Path):
    with open(requirement_file, 'r') as fp:
        reqs = [r.strip() for r in fp]
    return reqs


setup(name='movie-recommender',
      version='1.1.0',
      author='Andy Wang',
      author_email='andy80136@gmail.com',
      description='Recommender system in PyTorch',
      license='MIT License',
      packages=find_packages(),
      install_requires=parse_requirement('requirements/core.txt'),
      extras_require={'dev': parse_requirement('requirements/dev.txt')})
