"""
Configuration file!
"""

import logging
import warnings
import argparse

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Dataset Path
# =============================================================================
prefixPathCOCO = '/data1/dataset/MS-COCO_2014/'
prefixPathVG = '/data5/VG/'
prefixPathVOC2007 = '/data1/dataset/voc2007/VOCdevkit/VOC2007/'
# =============================================================================

# ClassNum of Dataset
# =============================================================================
_ClassNum = {'COCO2014': 80,
             'VOC2007': 20,
             'VG': 200,
            }
# =============================================================================


# Argument Parse
# =============================================================================
def str2bool(input):
    if isinstance(input, bool):
        return input

    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def show_args(args):

    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    for arg, content in args.__dict__.items():
        logger.info("{}: {}".format(arg, content))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")


def arg_parse(mode):

    assert mode in ('SST', 'SARB', 'HST', 'SARB-journal')

    parser = argparse.ArgumentParser(description='HCP Multi-label Image Recognition with Partial Labels')

    # Basic Augments
    parser.add_argument('--post', type=str, default='', help='postname of save model')
    parser.add_argument('--printFreq', type=int, default='1000', help='number of print frequency (default: 1000)')

    parser.add_argument('--mode', type=str, default='SST', choices=['SST', 'SARB', 'HST', 'SARB-journal'], help='mode of experiment (default: SST)')
    parser.add_argument('--dataset', type=str, default='COCO2014', choices=['COCO2014', 'VG', 'VOC2007'], help='dataset for training and testing')
    parser.add_argument('--prob', type=float, default=0.5, help='hyperparameter of label proportion (default: 0.5)')

    parser.add_argument('--pretrainedModel', type=str, default='None', help='path to pretrained model (default: None)')
    parser.add_argument('--resumeModel', type=str, default='None', help='path to resume model (default: None)')
    parser.add_argument('--evaluate', type=str2bool, default='False', help='whether to evaluate model (default: False)')

    parser.add_argument('--epochs', type=int, default=20, help='number of total epochs to run (default: 20)')
    parser.add_argument('--startEpoch', type=int, default=0, help='manual epoch number (default: 0)')
    parser.add_argument('--stepEpoch', type=int, default=10, help='decend the lr in epoch number (default: 10)')

    parser.add_argument('--batchSize', type=int, default=8, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weightDecay', type=float, default=1e-4, help='weight decay (default: 0.0001)')

    parser.add_argument('--cropSize', type=int, default=448, help='size of crop image')
    parser.add_argument('--scaleSize', type=int, default=512, help='size of rescale image')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')

    # Aguments for SST
    if mode == 'SST':
        parser.add_argument('--generateLabelEpoch', type=int, default=5, help='when to generate pseudo label (default: 5)')

        parser.add_argument('--intraBCEMargin', type=float, default=1.0, help='margin of intra bce loss (default: 1.0)')
        parser.add_argument('--intraBCEWeight', type=float, default=1.0, help='weight of intra bce loss (default: 1.0)')
        parser.add_argument('--intraCooccurrenceWeight', type=float, default=1.0, help='weight of intra co-occurrence loss (default: 1.0)')

        parser.add_argument('--interBCEMargin', type=float, default=1.0, help='margin of inter bce loss (default: 1.0)')
        parser.add_argument('--interBCEWeight', type=float, default=1.0, help='weight of inter bce loss (default: 1.0)')
        parser.add_argument('--interDistanceWeight', type=float, default=1.0, help='weight of inter Distance loss (default: 1.0)')
        parser.add_argument('--interExampleNumber', type=int, default=50, help='number of inter positive number (default: 50)')

    # Aguments for SARB
    if mode == 'SARB':
        parser.add_argument('--mixupEpoch', type=int, default=5, help='when to mix up (default: 5)')
        parser.add_argument('--contrastiveLossWeight', type=float, default=1.0, help='weight of contrastiveloss (default: 1.0)')
        
        parser.add_argument('--prototypeNum', type=int, default=10, help='number of the prototype (default: 10)')
        parser.add_argument('--recomputePrototypeInterval', type=int, default=5, help='interval of recomputing prototypes whose value greater than zero means recomputing prototypes (default: 5)')

        parser.add_argument('--isAlphaLearnable', type=str2bool, default='True', help='whether to set alpha be learnable  (default: True)')
        parser.add_argument('--isBetaLearnable', type=str2bool, default='True', help='whether to beta be learnable (default: True)')

    # Aguments for HST
    if mode == 'HST':
        parser.add_argument('--generateLabelEpoch', type=int, default=5, help='when to generate pseudo label (default: 5)')

        parser.add_argument('--intraMargin', type=float, default=1.0)
        parser.add_argument('--isIntraMarginLearnable', type=str2bool, default='True')        

        parser.add_argument('--intraBCEWeight', type=float, default=1.0)
        parser.add_argument('--intraCooccurrenceWeight', type=float, default=1.0, help='weight of intra co-occurrence loss (default: 1.0)')

        parser.add_argument('--interMargin', type=float, default=1.0)
        parser.add_argument('--isInterMarginLearnable', type=str2bool, default='True')

        parser.add_argument('--interBCEWeight', type=float, default=1.0)
        parser.add_argument('--interInstanceDistanceWeight', type=float, default=1.0, help='weight of inter Instance Distance loss (default: 1.0)')
        parser.add_argument('--interPrototypeDistanceWeight', type=float, default=1.0, help='weight of inter Prototype Distance loss (default: 1.0)')
    
        parser.add_argument('--prototypeNumber', type=int, default=50, help='number of inter positive number (default: 50)')
        parser.add_argument('--useRecomputePrototype', type=str2bool, default='False', help='whether to recompute prototype (default: False)')
        parser.add_argument('--computePrototypeEpoch', type=int, default=5, help='when to generate pseudo label (default: 5)')
        
    # Aguments for SARB-journal
    if mode == 'SARB-ournal':
        parser.add_argument('--mixupEpoch', type=int, default=5, help='when to mix up (default: 5)')
        parser.add_argument('--contrastiveLossWeight', type=float, default=1.0, help='weight of contrastiveloss (default: 1.0)')
        
        parser.add_argument('--topK', type=int, default=1, help='number of K (default: 1)')
        parser.add_argument('--recomputePrototypeInterval', type=int, default=5, help='interval of recomputing prototypes whose value greater than zero means recomputing prototypes (default: 5)')

        parser.add_argument('--isAlphaLearnable', type=str2bool, default='True', help='whether to set alpha be learnable  (default: True)')
        parser.add_argument('--isBetaLearnable', type=str2bool, default='True', help='whether to beta be learnable (default: True)')
        
    args = parser.parse_args()
    args.classNum = _ClassNum[args.dataset]    

    return args
# =============================================================================
