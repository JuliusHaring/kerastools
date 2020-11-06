from setuptools import setup

setup(name='kerastools',
      version='0.1',
      description='A keras wrapper',
      url='todo',
      author='Julius Haring',
      author_email='julius.h9191@googlemail.com',
      license='MIT',
      packages=['kerastools'],
      install_requires=[
          'tensorflow',
          'tensorboard',
          'sklearn',
      ],
      zip_safe=False)