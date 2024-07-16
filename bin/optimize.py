# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Execution controller
Depending on the environment, executes a MPI or multiprocessing version.
"""
import sys
import traceback
from feshdock.util.logger import LoggingManager
from feshdock.util.parser import CommandLineParser
import os,time


log = LoggingManager.get_logger("feshdock")


if __name__ == "__main__":

    try:
        parser = CommandLineParser()

        mpi_support = parser.args.mpi
        if mpi_support:
            from feshdock.simulation.docking_mpi import (
                run_simulation as mpi_simulation,
            )

            mpi_simulation(parser)
        else:
            from feshdock.simulation.docking_multiprocessing import (
                run_simulation as multiprocessing_simulation,
            )
            multiprocessing_simulation(parser)


    except Exception:
        log.error("feshdock has failed, please check traceback:")
        traceback.print_exc()
