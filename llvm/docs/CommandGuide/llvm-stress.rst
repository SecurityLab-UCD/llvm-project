llvm-stress - generate random .ll files
=======================================

.. program:: llvm-stress

SYNOPSIS
--------

:program:`llvm-stress` [-size=filesize] [-seed=initialseed] [-o=outfile]

DESCRIPTION
-----------

The :program:`llvm-stress` tool is used to randomly generate or mutate ``.ll``
or ``.bc`` files, which can be used to test different components of LLVM.

OPTIONS
-------

.. option:: -i filename

 Specify the input filename for mutating an existing module.

.. option:: -o filename

 Specify the output filename. Defaults to stdout.

.. option:: -seed seed

 Specify the seed to be used for the randomly generated instructions. If not
 specified, a seed will be generated.

.. option:: -repeat times

 Specify the number of times to mutate. The repeat count must be at
 least 1. Defaults to 100 times.

.. option:: -max-size bytes

 Specify the maximum bitcode size of the new module in bytes. The input module,
 if specified, must not have a greater bitcode size. Defaults to 1 MiB.

EXIT STATUS
-----------

:program:`llvm-stress` returns 0.
