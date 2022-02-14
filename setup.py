import setuptools
from os import path
import quickq

here = path.abspath(path.dirname(__file__))
AUTHORS = """
Stephanie Valleau
Evan Komp
"""


# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name='quickq',
        version=quickq.__version__,
        author=AUTHORS,
        project_urls={
            'Source': 'https://github.com/valleau-lab/quickq',
        },
        description=
        'Prediction of partition functions with trained ML predictor',
        long_description=long_description,
        include_package_data=False, #no data yet, True if we want to include data
        keywords=[
            'Machine Learning', 'Reaction Kinetics', 'Ab initio',
            'Chemical Engineering','Chemistry', 
        ],
        license='MIT',
        packages=setuptools.find_packages(exclude="tests"),
        scripts = [], #if we want to include shell scripts we make in the install
        install_requires=[
            'tensorflow',
            'numpy',
            'pandas',
            'ase',
            'molml',
            'dscribe',
        ],
        extras_require={
            'tests': [
                'pytest',
                'coverage',
                'flake8',
                'flake8-docstrings'
            ],
            'docs': [
                'sphinx',
                'sphinx_rtd_theme',

            ]
        },
        classifiers=[
            'Development Status :: 1 - Planning',
            'Environment :: Console',
            'Operating System :: OS Independant',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
        ],
        zip_safe=False,
    )
