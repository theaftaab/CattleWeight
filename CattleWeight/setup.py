import setuptools

setuptools.setup(include_package_data=True,
                 name='CattleWeight',
                 version='0.0.11',
                 description="Cattle Weight Estimator",
                 author="Aftaab Hussain",
                 author_email="aftaabhussaint@gmail.com",
                 package=setuptools.find_packages(),
                 install_requires=['pandas', 'numpy', 'opencv-python', 'ultralytics', 'joblib', 'scikit-learn'],
                 package_data={
                     'CattleWeight': ['*.pt', '*.csv', '*.txt',
                                      'requirements.txt', '*.pkl']
                 }
                 )
