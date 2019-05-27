import os

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext
import pkg_resources


from glob import glob
import tarfile
import shutil
import subprocess

# Not the greatest way to handle Python 2/3 changes
# but this avoids the need for the 'future' modules,
# which was causing some headaches in installation.
try:
    # Python 2
    from urllib import urlretrieve
except:
    try:
        # Python 3
        from urllib.request import urlretrieve
    except:
        raise Exception("Could not import urlretrieve.")


# Hold off on locating the numpy include directory 
# until we are actually building the extensions, by which 
# point numpy should have been installed
class build_ext(_build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')
        for ext in self.extensions:
            if hasattr(ext, 'include_dirs') and not numpy_incl in ext.include_dirs:
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)

extensions = [] # no extensions for now

setup(
    name='functionalmf',
    version='1.0',
    description='''Bayesian factorization of matrices where every entry is a curve or function rather than a scalar.''',
    author='Wesley Tansey',
    author_email='wes.tansey@gmail.com',
    url='http://www.github.com/tansey/functionalmf',
    license="GNU GPLv3",
    packages=['functionalmf'],
    ext_modules=extensions,
    install_requires=['numpy', 'scipy', 'matplotlib', 'pypolyagamma', 'seaborn', 'scikit-sparse'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python'
        ],
    keywords=['monte-carlo', 'lasso', 'trend-filtering', 'smoothing', 'tensor-factorization', 'matrix-factorization'],
    platforms="ALL",
    cmd_class = {'build_ext': build_ext}
)