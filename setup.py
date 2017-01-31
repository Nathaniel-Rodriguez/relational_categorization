from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='relcat',
    version='0.1',
    description='Uses Python 3.5 or later. Implements a relational categorization task used in (Williams, 2008) and (Williams, 2013).',
    author='Nathaniel Rodriguez',
    packages=['relcat'],
    url='https://github.com/Nathaniel-Rodriguez/relcat.git',
    install_requires=[
          'numpy',
          'matplotlib'
      ],
    include_package_data=True,
    zip_safe=False)