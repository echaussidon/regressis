#!/usr/bin/env python
# coding: utf-8

if __name__ == '__main__':

    import os
    import shutil

    input_dir = '../res'
    output_dir = '/global/cfs/cdirs/desi/users/edmondc/sys_weight/photo'
    engine = 'RF'
    dir_versions = [dir for dir in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, dir))]

    for v in dir_versions:
        dir_tracers = [dir for dir in os.listdir(os.path.join(input_dir, v)) if os.path.isdir(os.path.join(input_dir, v, dir)) and dir[0] != '.']

        for t in dir_tracers:

            tracer = t[len(v) + 1:-4]
            nside = t[-3:]

            shutil.copy(os.path.join(input_dir, v, t, engine, f'{v}_{tracer}_imaging_weight_{nside}.npy'), output_dir)
