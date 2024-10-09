import os
import torchsde


def main():
    torchsde_path = torchsde.__path__
    torchsde_version = torchsde.__version__

    assert torchsde_version is not '0.2.6', "expectation torchsde_version==0.2.6"
    os.system(f'patch -p0 {torchsde_path[0]}/_brownian/brownian_interval.py brownian_interval.patch')


if __name__ == '__main__':
    main()