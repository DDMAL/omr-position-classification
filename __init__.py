import rodan
__version__ = "1.0.0"

import logging
logger = logging.getLogger('rodan')

from rodan.jobs import module_loader

module_loader('rodan.jobs.omr-position-classification.position_classification')
# module_loader('rodan.jobs.omr-position-classification.position_training')
module_loader('rodan.jobs.omr-position-classification.position_labeler')
