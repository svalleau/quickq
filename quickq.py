"""Command line interface to the package"""
import argparse

import quickq.loader
import quickq.pipeline
import quickq.io
from quickq.features import EncodedBonds

parser = argparse.ArgumentParser(description='Predict partition functions.')
parser.add_argument(
    'root',
    type=str, 
    help='Path to root directory containing structure files or reaction directories.')
parser.add_argument(
    '-l',
    '--logfile',
    type='str',
    help='Specify path to file to log actions.'
)
parser.add_mutually_exclusive_group(required=True)
parser.add_argument(
    '-q', '--qest',
    action='store_true',
    help="Use Qest to predict partition functions of molecules"
)
parser.add_argument(
    '-t', '--qests',
    action='store_true',
    help="Use QesTS to predict partition functions of unknown transition states."
)
parser.add_argument(
    '-d', '--double',
    action='store_true',
    help="Use Qest and QesTS to predict partition functions of unknown transition states."
)
args = parser.parse_args()

if args.logfile:
    logger = logging.basicConfig(filename=args.logfile, level=logging.INFO)

if __name__ == '__main__':
    if args.qest:
        loader = quickq.loader.QestLoader(EncodedBonds)
        dataset = loader.create_dataset(root=args.root, data_dir='./.quickq_data/qest')
        y_hat = quickq.pipeline.predict_qest(dataset)
        quickq.io.save_Q_mols(args.root, dataset.ids, y_hat)