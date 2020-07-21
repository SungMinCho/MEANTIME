def find_saturation_point(df, wait_epochs, display=True):
    max_ndcg = -1
    max_epoch = -1
    reached_end = False
    if display:
        print('Finding saturation point')
    for epoch, ndcg in zip(df['epoch'], df['NDCG@10']):
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            max_epoch = epoch
        elif epoch - max_epoch >= wait_epochs:
            if display:
                print('Breaking because there was no improvement for the last {} epochs'.format(wait_epochs))
            break
    else:
        if display:
            print('Reached the end of experiment without saturation')
        reached_end = True
    if display:
        print('Saturation epoch={} ndcg@10={}'.format(max_epoch, max_ndcg))
    return df[df['epoch'] == max_epoch], reached_end
