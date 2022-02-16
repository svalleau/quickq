"""Command line interface to the package"""
import argparse

parser = argparse.ArgumentParser(description='Predict partition functions.')
parser.add_argument(
    'root',
    type=str, 
    help='Path to root directory containing structure files or reaction directories.')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    '-q', '--qest',
    action='store_true',
    help="Use Qest to predict partition functions of molecules"
)
group.add_argument(
    '-t', '--qests',
    action='store_true',
    help="Use QesTS to predict partition functions of unknown transition states."
)
group.add_argument(
    '-d', '--double',
    action='store_true',
    help="Use Qest and QesTS to predict partition functions of unknown transition states."
)
args = parser.parse_args()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import quickq.loader
import quickq.pipeline
import quickq.io
from quickq.features import EncodedBonds
import tempfile

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tempdir:
        if args.qest:
            loader = quickq.loader.QestLoader(EncodedBonds)
            dataset = loader.create_dataset(root=args.root, data_dir=f'{tempdir}/qest')
            y_hat = quickq.pipeline.predict_qest(dataset)
            quickq.io.save_Q_mols(args.root, dataset.ids, y_hat)
            
        elif args.qests:
            loader = quickq.loader.QesTSLoader(EncodedBonds)
            dataset = loader.create_dataset(root=args.root, data_dir=f'{tempdir}/qests')
            y_hat = quickq.pipeline.predict_qests(dataset)
            quickq.io.save_Q_rxns(args.root, dataset.ids, y_hat)
            
        elif args.double:
            loader = quickq.loader.DoubleLoader(EncodedBonds)
            dataset = loader.create_dataset(root=args.root, data_dir=f'{tempdir}/double')
            y_hat = quickq.pipeline.predict_qests(dataset)
            quickq.io.save_Q_rxns(args.root, dataset.ids, y_hat)