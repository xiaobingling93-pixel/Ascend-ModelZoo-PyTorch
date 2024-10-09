import os
import diffusers


def main():
    diffusers_path = diffusers.__path__
    diffusers_version = diffusers.__version__

    assert diffusers_version is not '0.30.0', "expectation diffusers==0.30.0"
    os.system(f'patch -p0 {diffusers_path[0]}/schedulers/scheduling_cosine_dpmsolver_multistep.py precision_scheduling_cosine_dpmsolver_multistep.patch')


if __name__ == '__main__':
    main()