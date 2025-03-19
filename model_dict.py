from models import complex_FFNO_2D, FNO_2D, FFNO_2D, complex_concatenation_FFNO_2D, complete_complex_FFNO_2D

def get_model(args):
    model_dict = {
        'FNO_2D': FNO_2D,
        'FFNO_2D': FFNO_2D,
        'complex_FFNO_2D': complex_FFNO_2D,
        'complex_concatenation_FFNO_2D': complex_concatenation_FFNO_2D,
        'complete_complex_FFNO_2D': complete_complex_FFNO_2D
    }
    
    return model_dict[args.model].Model(args).cuda()