from utils.parser import get_parser_with_args
from utils.metrics import FocalLoss, dice_loss

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0
    #print('target.shape')
    #print(target.shape)#(1,512,512)
    # gamma=0, alpha=None --> CE
    #print(len(predictions))
    focal = FocalLoss(gamma=0, alpha=None)
    for prediction in predictions:
        #print('prediction')
        #print(prediction.shape)
        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss

