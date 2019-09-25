import pytest
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def test_image_reduction():
    science_frame = aspired.ImageReduction('examples/lhs6328.list')
    science_frame.reduce()
