from models import encoder
from models import resnet
from models import ssl

REGISTERED_MODELS = {
    'sim-clr': ssl.SimCLR,
}
