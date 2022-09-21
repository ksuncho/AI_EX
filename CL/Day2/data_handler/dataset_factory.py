from data_handler.SineNShot import SineNShot

def get_dataset(args):
    name = args['dataset']
    if name == 'sine':
        return SineNShot(batchsz=args['tasknum'],
                         k_shot=args['k_spt'],
                         k_query=args['k_qry'])
