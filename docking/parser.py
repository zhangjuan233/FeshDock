import argparse


class SetupCommandLineParser(object):
    """Parses the command line of lightdock_setup"""

    def __init__(self, input_args=None):
        parser = argparse.ArgumentParser(prog="docking_main")

        # Receptor
        parser.add_argument(
            "receptor_pdb",
            help="Receptor structure PDB file",
            # type=valid_file,
            metavar="receptor_pdb_file",
        )
        # Ligand
        parser.add_argument(
            "ligand_pdb",
            help="Ligand structure PDB file",
            # type=valid_file,
            metavar="ligand_pdb_file",
        )
        if input_args:
            self.args = parser.parse_args(input_args)
        else:
            self.args = parser.parse_args()