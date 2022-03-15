from setuptools import setup

package_data = {
    "gaminet": [
        "lib/lib_ebm_native_win_x64.dll",
        "lib/lib_ebm_native_linux_x64.so",
        "lib/lib_ebm_native_mac_x64.dylib",
        "lib/lib_ebm_native_win_x64.pdb"
    ]
}

setup(name='gaminet',
      version='1.0.0',
      description='Pytorch version of GAMINet; it was done when I was PhD student in HKU',
      url='https://github.com/ZebinYang/GAMINet-Pytorch',
      author='Zebin Yang',
      author_email='yangzb2010@connect.hku.hk',
      license='GPL',
      packages=['gaminet'],
      package_data=package_data,
      install_requires=['matplotlib>=3.1.3', 'numpy>=1.15.2', 'pandas>=0.19.2', 'scikit-learn>=0.23.0', 'torch'],
      zip_safe=False)
