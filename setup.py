from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    """
    This function return the list of requirements from file_path
    """
    with open(file_path) as f:
        requirements = [line.strip() for line in f]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="CreditScoreProject",
    version="0.0.1",
    author="Almo",
    author_email="kulinanalmosenja@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)